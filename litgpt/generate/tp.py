"""Tensor-parallel implementation adapted from https://github.com/pytorch-labs/gpt-fast/blob/14df27/tp.py"""

import logging
import sys
import time
from functools import partial
from pathlib import Path
from typing import Literal, Optional, Union

import lightning as L
import torch
import torch._dynamo.config
import torch._inductor.config
from lightning.fabric.plugins import BitsandbytesPrecision
from lightning.fabric.utilities import rank_zero_only
from torch.distributed._functional_collectives import all_reduce

import litgpt.generate.base as generate_base
from litgpt import GPT, Config, Tokenizer
from litgpt.model import CausalSelfAttention, GptNeoxMLP, LLaMAMLP, LLaMAMoE
from litgpt.utils import CLI, check_valid_checkpoint_dir, get_default_supported_precision


def tensor_parallel_linear(fabric: L.Fabric, linear: torch.nn.Linear, style: str) -> None:
    world_size = fabric.world_size
    dim, attr = {"colwise": (0, "out_features"), "rowwise": (1, "in_features")}[style]
    size = getattr(linear, attr)
    if size % world_size != 0:
        raise ValueError(
            f"This linear's {attr} value ({size}) is not evenly divisible by the world size ({world_size})"
        )

    shard = torch.tensor_split(linear.weight, world_size, dim=dim)[fabric.global_rank]
    # overwrite `.data` instead of recreating the parameter for quantization (bitsandbytes) support.
    # the bitsandbytes linear classes use custom `torch.nn.Parameter` subclasses
    linear.weight.data = shard
    setattr(linear, attr, shard.size(dim))

    if linear.bias is not None and dim == 0:
        shard = torch.tensor_split(linear.bias, world_size)[fabric.global_rank]
        linear.bias = torch.nn.Parameter(shard, requires_grad=linear.bias.requires_grad)


def tensor_parallel_mlp(fabric: L.Fabric, mlp: Union[GptNeoxMLP, LLaMAMLP, LLaMAMoE]) -> None:
    if isinstance(mlp, LLaMAMLP):
        tensor_parallel_linear(fabric, mlp.fc_1, "colwise")
        tensor_parallel_linear(fabric, mlp.fc_2, "colwise")
        tensor_parallel_linear(fabric, mlp.proj, "rowwise")
        mlp.register_forward_hook(partial(all_reduce_output, fabric.world_size))
    elif isinstance(mlp, GptNeoxMLP):
        tensor_parallel_linear(fabric, mlp.fc, "colwise")
        tensor_parallel_linear(fabric, mlp.proj, "rowwise")
        mlp.register_forward_hook(partial(all_reduce_output, fabric.world_size))
    elif isinstance(mlp, LLaMAMoE):
        # we use expert slicing across ranks, alternatively, we could create a expert parallelism group
        # when the number of experts is a multiple of the world size
        for expert in mlp.experts:
            tensor_parallel_mlp(fabric, expert)
    else:
        raise NotImplementedError


def tensor_parallel_attn(fabric: L.Fabric, attn: CausalSelfAttention) -> None:
    tensor_parallel_linear(fabric, attn.attn, "colwise")
    tensor_parallel_linear(fabric, attn.proj, "rowwise")
    attn.register_forward_hook(partial(all_reduce_output, fabric.world_size))


def all_reduce_output(world_size: int, module: torch.nn.Module, ins, outs) -> torch.Tensor:
    return all_reduce(outs, "sum", list(range(world_size)))


def tensor_parallel(fabric: L.Fabric, model: GPT) -> GPT:
    for block in model.transformer.h:
        tensor_parallel_mlp(fabric, block.mlp)
        tensor_parallel_attn(fabric, block.attn)

    # update the config values to the shard sizes
    # this is only relevant for `tensor_parallel_attn`, but it needs to run only once
    world_size = fabric.world_size
    attrs = ["n_head", "n_embd", "n_query_groups"]
    for attr in attrs:
        size = getattr(model.config, attr)
        if size % world_size != 0:
            raise ValueError(f"This {attr} value ({size}) is not evenly divisible by the world size ({world_size})")
        setattr(model.config, attr, size // world_size)

    return model


@torch.inference_mode()
def main(
    prompt: str = "What food do llamas eat?",
    *,
    num_samples: int = 1,
    max_new_tokens: int = 50,
    top_k: Optional[int] = 200,
    temperature: float = 0.8,
    checkpoint_dir: Path = Path("checkpoints/stabilityai/stablelm-base-alpha-3b"),
    quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq"]] = None,
    precision: Optional[str] = None,
    compile: bool = False,
) -> None:
    """Generates text samples based on a pre-trained model and tokenizer.

    Args:
        prompt: The prompt string to use for generating the samples.
        num_samples: The number of text samples to generate.
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
        checkpoint_dir: The checkpoint directory to load.
        quantize: Whether to quantize the model and using which method:
            - bnb.nf4, bnb.nf4-dq, bnb.fp4, bnb.fp4-dq: 4-bit quantization from bitsandbytes
            for more details, see https://github.com/Lightning-AI/litgpt/blob/main/tutorials/quantize.md
        precision: Indicates the Fabric precision setting to use.
        compile: Whether to compile the model.
    """
    precision = precision or get_default_supported_precision(training=False)

    plugins = None
    if quantize is not None:
        if compile:
            raise NotImplementedError  # untested
        if "mixed" in precision:
            raise ValueError("Quantization and mixed precision is not supported.")
        dtype = {"16-true": torch.float16, "bf16-true": torch.bfloat16, "32-true": torch.float32}[precision]
        bnb_logger = logging.getLogger("lightning.fabric.plugins.precision.bitsandbytes")
        bnb_logger.setLevel(logging.DEBUG)
        bnb_logger.debug = rank_zero_only(bnb_logger.debug)
        plugins = BitsandbytesPrecision(quantize[4:], dtype)
        precision = None

    # set "ddp" as the strategy for the launching functionality, but there's no data-parallelism
    fabric = L.Fabric(devices="auto", strategy="ddp", precision=precision, plugins=plugins)
    fabric.launch()

    check_valid_checkpoint_dir(checkpoint_dir)

    config = Config.from_file(checkpoint_dir / "model_config.yaml")

    model_file = "lit_model.pth"
    checkpoint_path = checkpoint_dir / model_file

    tokenizer = Tokenizer(checkpoint_dir)
    encoded = tokenizer.encode(prompt, device=fabric.device)
    prompt_length = encoded.size(0)
    max_returned_tokens = prompt_length + max_new_tokens

    fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}", file=sys.stderr)
    t0 = time.perf_counter()
    # cannot use `init_module` because if bitsandbytes is used, the Linear layers will be replaced
    # which means that the weights will get quantized on cuda:0 on checkpoint load. we need to load and then convert
    # still, use init_tensor for the precision
    with fabric.init_tensor(), torch.device("meta"):
        model = GPT(config)
    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    # sequentially do: load the checkpoint on CPU -> quantize -> apply tp -> move to device
    # so that the CPU RAM doesn't OOM with larger models
    for rank in range(fabric.world_size):
        if fabric.global_rank == rank:
            t0 = time.perf_counter()
            state_dict = torch.load(str(checkpoint_path), mmap=True, map_location="cpu")
            model.load_state_dict(state_dict, assign=True)
            print(f"[{rank}] Time to load the model weights: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

            # cannot use `.setup_module` because it will wrap with DDP
            model = fabric._precision.convert_module(model)

            t0 = time.perf_counter()
            model = tensor_parallel(fabric, model)
            print(
                f"[{rank}] Time to tensor-parallelize the model: {time.perf_counter() - t0:.02f} seconds.",
                file=sys.stderr,
            )

            with fabric.init_tensor():
                # set the max_seq_length to limit the memory usage to what we need
                model.max_seq_length = max_returned_tokens
                # the rope cache which is on meta device
                model.cos, model.sin = model.rope_cache()
                # enable the kv cache
                model.set_kv_cache(batch_size=1)
            model.eval()

            t0 = time.perf_counter()
            model = fabric.to_device(model)
            print(f"[{rank}] Time to move the model: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)
        fabric.barrier()

    if compile:
        torch._dynamo.config.automatic_dynamic_shapes = True
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.coordinate_descent_tuning = True
        generate_base.next_token = torch.compile(generate_base.next_token, mode="reduce-overhead")

    L.seed_everything(1234)
    for i in range(num_samples):
        t0 = time.perf_counter()
        y = generate_base.generate(
            model, encoded, max_returned_tokens, temperature=temperature, top_k=top_k, eos_id=tokenizer.eos_id
        )
        t = time.perf_counter() - t0
        for block in model.transformer.h:
            block.attn.kv_cache.reset_parameters()
        fabric.print(tokenizer.decode(y))
        tokens_generated = y.size(0) - prompt_length
        fabric.print(
            f"Time for inference {i + 1}: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec", file=sys.stderr
        )
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB", file=sys.stderr)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    CLI(main)

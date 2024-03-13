# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import itertools
import logging
import re
import sys
import time
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Literal, Optional

import lightning as L
import torch
from lightning.fabric.accelerators import CUDAAccelerator
from lightning.fabric.plugins import BitsandbytesPrecision
from lightning.fabric.utilities.init import _materialize_meta_tensors
from typing_extensions import Type

import litgpt.generate.base as generate_base
from litgpt import GPT, Config, Tokenizer
from litgpt.model import Block, build_mask_cache
from litgpt.utils import CLI, check_valid_checkpoint_dir, get_default_supported_precision


@torch.inference_mode()
def sequential(model: GPT, root: torch.device, max_seq_length: int, devices: int):
    if model.config.n_layer % devices:
        # TODO: support smarter partitioning schemes
        raise NotImplementedError(
            f"Only balanced partitioning is implemented: n_layer={model.config.n_layer}, devices {devices}"
        )
    layers_per_rank = model.config.n_layer // devices
    # dictates where each block should be instantiated
    mapping = layer_to_device(model, chunk_on=Block, chunk_size=layers_per_rank)

    # materialize each block on the appropriate device
    for path, target_index in mapping.items():
        submodule = model.get_submodule(path)
        target_device = torch.device(root.type, target_index)
        print(f"Moving {path!r} to {target_device}", file=sys.stderr)
        # submodules loaded by the checkpoint will be on CPU (if no quantization). move them
        replace_device(submodule, replace=torch.device("cpu"), by=target_device)
        # in case the checkpoint was partial, materialize leftover metas
        _materialize_meta_tensors(submodule, target_device)
        # and build the kv cache
        submodule.attn.kv_cache = submodule.attn.build_kv_cache(1, max_seq_length, model.cos.size(-1), target_device)
    # rebuild odd ends
    with root:
        model.max_seq_length = max_seq_length
        # the rope cache which is on meta device
        model.cos, model.sin = model.rope_cache()
        # the mask cache which cannot be created with `set_kv_cache` because that will set it for all layers
        model.mask_cache = build_mask_cache(max_seq_length)
    # and everything that is not a block in the root
    _materialize_meta_tensors(model, root)
    replace_device(model, replace=torch.device("cpu"), by=root)

    if devices > 1:
        # install hooks to move layer inputs/output between devices
        for layer_num, (path, target_index) in enumerate(mapping.items()):
            submodule = model.get_submodule(path)
            if layer_num >= layers_per_rank:
                # we need to move the block input on the boundaries between devices
                # and also on every non-root device because the RoPE and mask cache is shared
                # TODO: the second case could be optimized and then we would only need this hook for
                # `layer_num in [layers_per_rank * i - 1 for i in range(1, devices + 1)]`
                target_device = torch.device(root.type, target_index)
                submodule.register_forward_pre_hook(partial(move_block_input, target_device))
            if layer_num == model.config.n_layer - 1:
                submodule.register_forward_hook(partial(move_block_output, root))

    return model


def layer_to_device(
    module: torch.nn.Module, chunk_on: Type[torch.nn.Module], chunk_size: int
) -> "OrderedDict[str, int]":
    """Create a mapping from layer (block) to device."""
    # this assumes that the definition order is the same as the execution order
    hits = [name for name, submodule in module.named_modules() if isinstance(submodule, chunk_on)]
    return OrderedDict((name, i // chunk_size) for i, name in enumerate(hits))


def move_block_input(device: torch.device, module: torch.nn.Module, ins):
    """``forward_pre_hook`` to move a Block's input before forward."""
    # during inference, none of the inputs are None: x, cos, sin, mask, input_pos
    return tuple(t.to(device) for t in ins)


def move_block_output(device: torch.device, module: torch.nn.Module, ins, outs) -> torch.Tensor:
    """``forward_hook`` to move a Block's output after forward."""
    return outs.to(device)


def replace_device(module: torch.nn.Module, replace: torch.device, by: torch.device) -> torch.nn.Module:
    for name, submodule in module.named_modules():
        tensors = dict(
            itertools.chain(submodule.named_parameters(recurse=False), submodule.named_buffers(recurse=False))
        )
        if not tensors:
            continue
        devices = {t.device for t in tensors.values()}
        if len(devices) != 1:
            # since this is using `submodule.to`, different devices in the same submodule is a problem
            path_to_device = {f"{name}.{p}": t.device for p, t in tensors.items()}
            raise ValueError(f"Found multiple devices: {path_to_device}")
        if devices.pop() == replace:
            submodule.to(by)
    return module


@torch.inference_mode()
def main(
    prompt: str = "What food do llamas eat?",
    *,
    num_samples: int = 1,
    max_new_tokens: int = 50,
    top_k: Optional[int] = 200,
    temperature: float = 0.8,
    checkpoint_dir: Path = Path("checkpoints/mistralai/Mistral-7B-Instruct-v0.1"),
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
        logging.getLogger("lightning.fabric.plugins.precision.bitsandbytes").setLevel(logging.DEBUG)
        plugins = BitsandbytesPrecision(quantize[4:], dtype)
        precision = None

    fabric = L.Fabric(devices=1, precision=precision, accelerator="cuda", plugins=plugins)

    total_devices = CUDAAccelerator.auto_device_count()
    print(f"Using {total_devices} devices", file=sys.stderr)

    check_valid_checkpoint_dir(checkpoint_dir)

    config = Config.from_file(checkpoint_dir / "model_config.yaml")

    checkpoint_path = checkpoint_dir / "lit_model.pth"

    tokenizer = Tokenizer(checkpoint_dir)
    encoded = tokenizer.encode(prompt, device=fabric.device)
    prompt_length = encoded.size(0)
    max_returned_tokens = prompt_length + max_new_tokens

    print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}", file=sys.stderr)
    t0 = time.perf_counter()
    # cannot use `init_module` because if bitsandbytes is used, the Linear layers will be replaced
    # which means that the weights will get quantized on cuda:0 on checkpoint load. we need to load and then convert
    # still, use init_tensor for the precision
    with fabric.init_tensor(), torch.device("meta"):
        model = GPT(config)
    print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    t0 = time.perf_counter()
    state_dict = torch.load(str(checkpoint_path), mmap=True, map_location="cpu")
    # TODO: this assumes that the model fits on CPU. Use lazy_load and make the materialization checkpoint aware
    model.load_state_dict(state_dict, assign=True)
    print(f"Time to load the model weights: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    model = fabric.setup_module(model, move_to_device=False)

    t0 = time.perf_counter()
    model = sequential(model, fabric.device, max_returned_tokens, total_devices)
    print(f"Time to sequential-ize the model: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    if compile:
        # TODO: raises an internal compile AssertionError caused by fabric.strategy.precision.forward_context
        raise NotImplementedError
        # silence developer warning on nightly builds
        # https://github.com/pytorch/pytorch/blob/v2.2.0-rc5/torch/_inductor/ir.py#L4166
        pattern = re.compile(".*DeviceCopy in input program.*")
        logging.getLogger("torch._inductor.utils").addFilter(lambda record: not pattern.search(record.getMessage()))
        torch._dynamo.config.automatic_dynamic_shapes = True
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.coordinate_descent_tuning = True
        # cannot use cudagraphs because it doesn't support multiple device indices
        # https://github.com/pytorch/pytorch/blob/v2.2.0-rc5/torch/_inductor/compile_fx.py#L371-L375
        generate_base.next_token = torch.compile(generate_base.next_token)

    L.seed_everything(1234)
    for i in range(num_samples):
        t0 = time.perf_counter()
        y = generate_base.generate(
            model, encoded, max_returned_tokens, temperature=temperature, top_k=top_k, eos_id=tokenizer.eos_id
        )
        t = time.perf_counter() - t0
        for block in model.transformer.h:
            block.attn.kv_cache.reset_parameters()
        print(tokenizer.decode(y))
        tokens_generated = y.size(0) - prompt_length
        print(
            f"Time for inference {i + 1}: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec", file=sys.stderr
        )
    print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB", file=sys.stderr)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    CLI(main)

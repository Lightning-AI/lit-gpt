# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Literal, Optional

import lightning as L
import torch
import torch._dynamo.config
import torch._inductor.config
from lightning.fabric.plugins import BitsandbytesPrecision

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt import GPT, Config, Tokenizer
from lit_gpt.utils import CLI, check_valid_checkpoint_dir, get_default_supported_precision, load_checkpoint


def multinomial_num_samples_1(probs: torch.Tensor) -> torch.Tensor:
    if torch._dynamo.is_compiling():
        # Faster alternative to `torch.multinomial(probs, num_samples=1)` that is also CUDAGraph friendly
        distribution = torch.empty_like(probs).exponential_(1)
        return torch.argmax(probs / distribution, dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1)


def sample(logits: torch.Tensor, temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
    logits = logits[0, -1]
    # optionally crop the logits to only the top k options
    if top_k is not None:
        v, i = torch.topk(logits, min(top_k, logits.size(-1)))
        # do not use `torch.where` as in nanogpt because it will repeat top-k collisions
        logits = torch.full_like(logits, float("-inf")).scatter_(-1, i, v)
    # optionally scale the logits and sample from a probability distribution
    if temperature > 0.0:
        probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
        return multinomial_num_samples_1(probs)
    return torch.argmax(logits, dim=-1, keepdim=True)


def next_token(model: GPT, input_pos: torch.Tensor, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    logits = model(x, input_pos)
    next = sample(logits, **kwargs)
    return next.to(dtype=x.dtype)


@torch.inference_mode()
def generate(
    model: GPT,
    prompt: torch.Tensor,
    max_returned_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    eos_id: Optional[int] = None,
) -> torch.Tensor:
    """Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.

    The implementation of this function is modified from A. Karpathy's nanoGPT.

    Args:
        model: The model to use.
        prompt: Tensor of shape (T) with indices of the prompt sequence.
        max_returned_tokens: The maximum number of tokens to return (given plus generated).
        temperature: Scales the predicted logits by 1 / temperature.
        top_k: If specified, only sample among the tokens with the k highest probabilities.
        eos_id: If specified, stop generating any more token once the <eos> token is triggered.
    """
    T = prompt.size(0)
    assert max_returned_tokens > T
    if model.max_seq_length < max_returned_tokens - 1:
        # rolling the kv cache based on the `input_pos` value would be necessary. However, doing so would introduce a
        # data dependency on the `input_pos` tensor and impact model compilation. Since this setting is uncommon, we do
        # not support it to avoid negatively impacting the overall speed
        raise NotImplementedError(f"max_seq_length {model.max_seq_length} needs to be >= {max_returned_tokens - 1}")

    device = prompt.device
    tokens = [prompt]
    input_pos = torch.tensor([T], device=device)
    token = next_token(
        model, torch.arange(0, T, device=device), prompt.view(1, -1), temperature=temperature, top_k=top_k
    ).clone()
    tokens.append(token)
    for _ in range(2, max_returned_tokens - T + 1):
        token = next_token(model, input_pos, token.view(1, -1), temperature=temperature, top_k=top_k).clone()
        tokens.append(token)
        if token == eos_id:
            break
        input_pos = input_pos.add_(1)
    return torch.cat(tokens)


@torch.inference_mode()
def main(
    prompt: str = "What food do llamas eat?",
    *,
    num_samples: int = 1,
    max_new_tokens: int = 50,
    top_k: Optional[int] = 200,
    temperature: float = 0.8,
    checkpoint_dir: Path = Path("checkpoints/stabilityai/stablelm-base-alpha-3b"),
    quantize: Optional[
        Literal[
            "bnb.nf4",
            "bnb.nf4-dq",
            "bnb.fp4",
            "bnb.fp4-dq",
            "bnb.int8",
            "gptq.int2",
            "gptq.int3",
            "gptq.int4",
            "gptq.int8",
        ]
    ] = None,
    gptq_kernel: Optional[Literal["cuda_old", "cuda", "exllama", "exllamav2", "triton", "marlin"]] = None,
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
            - bnb.int8: 8-bit quantization from bitsandbytes
            - gptq.int[bits]: inference with AutoGPTQ. Select the same `bits` value as was used during quantization.
            for more details, see https://github.com/Lightning-AI/lit-gpt/blob/main/tutorials/quantize.md
        gptq_kernel: Choose a kernel to apply with the quantized weights. If set to None, the same kernel used during
            quantization will be selected.
        precision: Indicates the Fabric precision setting to use.
        compile: Whether to compile the model.
    """
    precision = precision or get_default_supported_precision(training=False)

    plugins = None
    gptq_selected = False
    if quantize is not None and quantize.startswith("bnb."):
        if "mixed" in precision:
            raise ValueError("Quantization and mixed precision is not supported.")
        dtype = {"16-true": torch.float16, "bf16-true": torch.bfloat16, "32-true": torch.float32}[precision]
        plugins = BitsandbytesPrecision(quantize[4:], dtype)
        precision = None
    elif quantize is not None and quantize.startswith("gptq."):
        gptq_selected = True
        bits = quantize[-1]
        if precision != "16-true":
            print(
                f"AutoGPTQ requires float16 precision, but {precision} was selected. Overriding precision to float16."
            )
        precision = "16-true"

    fabric = L.Fabric(devices=1, precision=precision, plugins=plugins)

    # Config
    check_valid_checkpoint_dir(checkpoint_dir)
    config = Config.from_json(checkpoint_dir / "lit_config.json")

    if gptq_selected:
        from quantize.autogptq import QuantizeConfig

        quantized_model_dir = checkpoint_dir / f"quantized/{bits}bit"
        quantize_config = QuantizeConfig.load_config(quantized_model_dir / "quantize_config.json")
        gptq_kernel = gptq_kernel or quantize_config.kernel
        if gptq_kernel == "marlin" and quantize_config.marlin_cached:
            model_file = "marlin_cache.pth"
        else:
            model_file = "lit_model_gptq.pth"
        if not (quantized_model_dir / model_file).is_file():
            raise ValueError(f"`{model_file}` is missing. Please run `python quantize/autogptq.py` first.")
        checkpoint_path = quantized_model_dir / model_file
    else:
        checkpoint_path = checkpoint_dir / "lit_model.pth"

    # Tokenizer and model
    tokenizer = Tokenizer(checkpoint_dir)
    encoded = tokenizer.encode(prompt, device=fabric.device)
    prompt_length = encoded.size(0)
    max_returned_tokens = prompt_length + max_new_tokens

    fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}", file=sys.stderr)
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=True), torch.device("meta") if gptq_selected else nullcontext():
        model = GPT(config)
    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    with fabric.init_tensor():
        # set the max_seq_length to limit the memory usage to what we need
        model.max_seq_length = max_returned_tokens
        # enable the kv cache
        model.set_kv_cache(batch_size=1)
    model.eval()

    # GPTQ model conversion
    if gptq_selected:
        from quantize.autogptq import AutoGPTQ

        autogptq = AutoGPTQ(model=model, quantized=True, quantize_config=quantize_config)
        # If Marlin is selected and was cached - convert directly to Marlin (that allows to load cached weights)
        # If it wasn't cached - first convert to kernel from the config, load weights (in the code down below) and
        # the conversion to Marlin will be done later (in `convert_quantized_to_marlin` method)
        if (_gptq_kernel := gptq_kernel) == "marlin" and not quantize_config.marlin_cached:
            _gptq_kernel = quantize_config.kernel
        autogptq.convert_to_quantized(_gptq_kernel, fabric.device)

    # Loading weights
    t0 = time.perf_counter()
    if gptq_selected:
        state_dict = torch.load(str(checkpoint_path), mmap=True, map_location=fabric.device)
        model.load_state_dict(state_dict, assign=True)
        model.reset_parameters()
        model = fabric.setup_module(model)
    else:
        model = fabric.setup_module(model)
        load_checkpoint(fabric, model, checkpoint_path)
    fabric.print(f"Time to load the model weights: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    # Marlin conversion and post_init
    if gptq_selected:
        # Marlin conversion happens after the model is quantized to one of the other kernels
        if gptq_kernel == "marlin":
            autogptq.convert_quantized_to_marlin(quantized_model_dir)
        # obligatory post init: initializes kernel's buffers
        autogptq.post_init()
        # the last step is to print quantize config with the selected kernel
        quantize_config.kernel = gptq_kernel
        fabric.print(f"Running GPTQ model with {quantize_config}", file=sys.stderr)

    if compile:
        if gptq_selected:
            raise ValueError("GPTQ quantization and model compilation aren't supported.")
        torch._dynamo.config.automatic_dynamic_shapes = True
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.coordinate_descent_tuning = True
        global next_token
        next_token = torch.compile(next_token, mode="reduce-overhead")

    L.seed_everything(1234)
    for i in range(num_samples):
        t0 = time.perf_counter()
        y = generate(model, encoded, max_returned_tokens, temperature=temperature, top_k=top_k, eos_id=tokenizer.eos_id)
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

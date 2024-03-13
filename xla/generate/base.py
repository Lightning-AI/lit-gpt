# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import sys
import time
from pathlib import Path
from typing import Optional

import lightning as L
import torch
import torch_xla.core.xla_model as xm
from lightning.fabric.accelerators import XLAAccelerator
from lightning.fabric.strategies import XLAFSDPStrategy

from litgpt import GPT, Config, Tokenizer
from litgpt.model import Block
from litgpt.utils import check_valid_checkpoint_dir, lazy_load

# support running without installing as a package
wd = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(wd))

from xla.utils import rank_print


# xla does not support `inference_mode`: RuntimeError: Cannot set version_counter for inference tensor
@torch.no_grad()
def generate(
    model: GPT,
    idx: torch.Tensor,
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
        idx: Tensor of shape (T) with indices of the prompt sequence.
        max_returned_tokens: The maximum number of tokens to return (given plus generated).
        temperature: Scales the predicted logits by 1 / temperature.
        top_k: If specified, only sample among the tokens with the k highest probabilities.
        eos_id: If specified, stop generating any more token once the <eos> token is triggered.
    """
    T = idx.size(0)
    assert max_returned_tokens > T
    if model.max_seq_length < max_returned_tokens - 1:
        # rolling the kv cache based on the `input_pos` value would be necessary. However, doing so would introduce a
        # data dependency on the `input_pos` tensor and impact model compilation. Since this setting is uncommon, we do
        # not support it to avoid negatively impacting the overall speed
        raise NotImplementedError(f"max_seq_length {model.max_seq_length} needs to be >= {max_returned_tokens - 1}")

    device, dtype = idx.device, idx.dtype
    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty(max_returned_tokens, dtype=dtype, device=device)
    empty[:T] = idx
    idx = empty
    # TODO: FSDP has an internal broadcasting issue, so we are forced to have this be of length 1 until it's fixed
    input_pos = torch.tensor([0], device=device)

    xm.mark_step()

    # generate up to a fixed number of tokens
    for _ in range(max_returned_tokens):
        x = idx.index_select(0, input_pos).view(1, -1)

        # forward
        logits = model(x, input_pos)
        logits = logits[0, -1] / temperature

        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = torch.where(logits < v[[-1]], -float("Inf"), logits)

        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1).to(dtype=dtype)

        # advance
        input_pos = input_pos[-1:] + 1

        xm.mark_step()

        # concatenate the new generation
        idx = idx.index_copy(0, input_pos, idx_next)

        # if <eos> token is triggered, return the output (stop generation)
        if idx_next == eos_id:
            return idx[:input_pos]  # include the EOS token

    return idx


def setup(
    prompt: str = "What food do lamas eat?",
    *,
    num_samples: int = 1,
    max_new_tokens: int = 100,
    top_k: Optional[int] = 200,
    temperature: float = 0.8,
    checkpoint_dir: Path = Path("checkpoints/tiiuae/falcon-7b"),
    precision: str = "bf16-true",
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
        precision: Indicates the Fabric precision setting to use.
    """
    devices = XLAAccelerator.auto_device_count()
    strategy = XLAFSDPStrategy(auto_wrap_policy={Block}) if devices > 1 else "auto"
    fabric = L.Fabric(devices=devices, precision=precision, strategy=strategy)
    fabric.launch(main, prompt, num_samples, max_new_tokens, top_k, temperature, checkpoint_dir)


def main(
    fabric: L.Fabric,
    prompt: str,
    num_samples: int,
    max_new_tokens: int,
    top_k: Optional[int],
    temperature: float,
    checkpoint_dir: Path,
) -> None:
    check_valid_checkpoint_dir(checkpoint_dir)

    config = Config.from_file(checkpoint_dir / "model_config.yaml")

    checkpoint_path = checkpoint_dir / "lit_model.pth"

    rank_print(fabric, f"Loading model {str(checkpoint_path)!r} with {config.__dict__}", file=sys.stderr)
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=True):
        model = GPT(config)
    rank_print(fabric, f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    t0 = time.perf_counter()
    checkpoint = lazy_load(checkpoint_path)
    model.load_state_dict(checkpoint.get("model", checkpoint))
    rank_print(fabric, f"Time to load the model weights: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()
    model = fabric.setup_module(model)

    tokenizer = Tokenizer(checkpoint_dir)
    encoded = tokenizer.encode(prompt, device=fabric.device)
    prompt_length = encoded.size(0)
    max_returned_tokens = prompt_length + max_new_tokens

    with fabric.init_tensor():
        # set the max_seq_length to limit the memory usage to what we need
        model.max_seq_length = max_returned_tokens

    L.seed_everything(1234)
    for i in range(num_samples):
        with fabric.init_tensor():
            # enable the kv cache
            model.set_kv_cache(batch_size=1)

        t0 = time.perf_counter()
        y = generate(model, encoded, max_returned_tokens, temperature=temperature, top_k=top_k)
        t = time.perf_counter() - t0

        fabric.print(tokenizer.decode(y))
        tokens_generated = y.size(0) - prompt_length
        rank_print(
            fabric,
            f"Time for inference {i + 1}: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec",
            file=sys.stderr,
        )


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(setup)

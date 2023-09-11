import sys
import time
from pathlib import Path

import lightning as L
from lightning.fabric.accelerators import XLAAccelerator
from lightning.fabric.strategies import XLAFSDPStrategy

# support running without installing as a package
wd = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt import Tokenizer
from lit_gpt.adapter import GPT, Block, Config
from lit_gpt.utils import check_valid_checkpoint_dir, lazy_load
from scripts.prepare_alpaca import generate_prompt
from xla.generate.base import generate
from xla.utils import rank_print


def setup(
    prompt: str = "What food do lamas eat?",
    *,
    input: str = "",
    adapter_path: Path = Path("out/adapter/alpaca/lit_model_adapter_finetuned.pth"),
    checkpoint_dir: Path = Path("checkpoints/tiiuae/falcon-7b"),
    max_new_tokens: int = 100,
    top_k: int = 200,
    temperature: float = 0.8,
    precision: str = "bf16-true",
):
    """Generates a response based on a given instruction and an optional input.
    This script will only work with checkpoints from the instruction-tuned GPT-Adapter model.
    See `xla/finetune/adapter.py`.

    Args:
        prompt: The prompt/instruction (Alpaca style).
        input: Optional input (Alpaca style).
        adapter_path: Path to the checkpoint with trained adapter weights, which are the output of
            `finetune/adapter.py`.
        checkpoint_dir: The path to the checkpoint folder with pretrained GPT weights.
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
        precision: Indicates the Fabric precision setting to use.
    """
    devices = XLAAccelerator.auto_device_count()
    strategy = XLAFSDPStrategy(auto_wrap_policy={Block}) if devices > 1 else "auto"
    fabric = L.Fabric(devices=devices, precision=precision, strategy=strategy)
    fabric.launch(main, prompt, input, adapter_path, checkpoint_dir, max_new_tokens, top_k, temperature)


def main(
    fabric: L.Fabric,
    prompt: str,
    input: str,
    adapter_path: Path,
    checkpoint_dir: Path,
    max_new_tokens: int,
    top_k: int,
    temperature: float,
) -> None:
    check_valid_checkpoint_dir(checkpoint_dir)

    config = Config.from_json(checkpoint_dir / "lit_config.json", adapter_start_layer=0)

    checkpoint_path = checkpoint_dir / "lit_model.pth"

    rank_print(fabric, f"Loading model {str(checkpoint_path)!r} with {config.__dict__}", file=sys.stderr)
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=True):
        model = GPT(config)
    rank_print(fabric, f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    t0 = time.perf_counter()
    with lazy_load(checkpoint_path) as checkpoint, lazy_load(adapter_path) as adapter_checkpoint:
        checkpoint.update(adapter_checkpoint.get("model", adapter_checkpoint))
        model.load_state_dict(checkpoint)
    rank_print(fabric, f"Time to load the model weights: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()
    model = fabric.setup_module(model)

    tokenizer = Tokenizer(checkpoint_dir)
    sample = {"instruction": prompt, "input": input}
    prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt, device=fabric.device)
    prompt_length = encoded.size(0)
    max_returned_tokens = prompt_length + max_new_tokens

    with fabric.init_tensor():
        # set the max_seq_length to limit the memory usage to what we need
        model.max_seq_length = max_returned_tokens

    t0 = time.perf_counter()
    y = generate(
        model,
        encoded,
        max_returned_tokens,
        max_seq_length=max_returned_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_id=tokenizer.eos_id,
    )
    t = time.perf_counter() - t0

    model.reset_cache()
    output = tokenizer.decode(y)
    output = output.split("### Response:")[1] if "### Response:" in output else output
    output = output.strip()
    fabric.print(output)

    tokens_generated = y.size(0) - prompt_length
    rank_print(
        fabric, f"\n\nTime for inference: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec", file=sys.stderr
    )


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(setup)

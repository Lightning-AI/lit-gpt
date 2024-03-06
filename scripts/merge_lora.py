# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""This script merges the LoRA weights with the base model"""
import shutil
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import yaml

import lightning as L
import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt.lora import GPT, Config, lora_filter, merge_lora_weights
from lit_gpt.utils import CLI, check_valid_checkpoint_dir, get_default_supported_precision, lazy_load


def merge_lora(
    checkpoint_dir: Path = Path("out/finetune/lora/final"),
    pretrained_checkpoint_dir: Optional[Path] = None,
    precision: Optional[str] = None,
) -> None:
    """Merges the LoRA weights with the base model. See `finetune/lora.py`.

    Merging happens in-place in the checkpoint directory that is given as input.

    Args:
        checkpoint_dir: Path to the checkpoint directory with trained LoRA weights, which is the output of
            `finetune/lora.py`.
        pretrained_checkpoint_dir: Optional path to the checkpoint directory with the weights of the base model
            corresponding to the LoRA checkpoint. By default, this will automatically be inferred from the metadata
            in the given `checkpoint_dir` directory. Only set this if the base model checkpoint directory
            has moved or was renamed.
        precision: Optional precision setting to instantiate the model weights in. By default, this will
            automatically be inferred from the metadata in the given `checkpoint_dir` directory.
    """
    check_valid_checkpoint_dir(checkpoint_dir)
    if pretrained_checkpoint_dir is not None:
        check_valid_checkpoint_dir(pretrained_checkpoint_dir)
    if (checkpoint_dir / "lit_model.pth.lora").is_file():
        print("LoRA weights have already been merged in this checkpoint.")
        return

    lora_params, pretrained_checkpoint_dir, lora_precision = load_lora_metadata(checkpoint_dir)
    precision = precision if precision is not None else lora_precision

    fabric = L.Fabric(devices=1, precision=precision)
    config = Config.from_json(checkpoint_dir / "lit_config.json", **lora_params)

    with fabric.init_module(empty_init=True):
        model = GPT(config)

    lora_path = checkpoint_dir / "lit_model.pth"
    pretrained_checkpoint = lazy_load(pretrained_checkpoint_dir / "lit_model.pth")
    lora_checkpoint = lazy_load(lora_path)

    # Merge LoRA weights into the base model
    pretrained_checkpoint.update(lora_checkpoint.get("model", lora_checkpoint))
    model.load_state_dict(pretrained_checkpoint)
    merge_lora_weights(model)

    # Remove LoRA parameters and the LoRA linear substring
    state_dict = {k.replace("linear.", ""): v for k, v in model.state_dict().items() if not lora_filter(k, v)}
    save_path = checkpoint_dir / "lit_model.pth.merged"
    torch.save(state_dict, save_path)

    # Make a backup of the LoRA weights (they are only a few MBs)
    shutil.move(checkpoint_dir / "lit_model.pth", checkpoint_dir / "lit_model.pth.lora")
    shutil.move(checkpoint_dir / "lit_model.pth.merged", checkpoint_dir / "lit_model.pth")

    fabric.print(f"Saved merged weights to {str(checkpoint_dir / 'lit_model.pth')!r}")
    fabric.print(f"A backup of the old LoRA weights is in {str(checkpoint_dir / 'lit_model.pth.lora')!r}")


def load_lora_metadata(checkpoint_dir: Path) -> Tuple[Dict[str, Any], Path, str]:
    hparams_file = checkpoint_dir / "hyperparameters.yaml"
    if not hparams_file.is_file():
        raise FileNotFoundError()  # TODO

    with open(hparams_file, "r") as file:
        hparams = yaml.safe_load(file)

    lora_params = {k: v for k, v in hparams.items() if k.startswith("lora_")}
    pretrained_checkpoint_dir = Path(hparams["checkpoint_dir"])
    precision = hparams.get("precision")
    return lora_params, pretrained_checkpoint_dir, precision


if __name__ == "__main__":
    CLI(merge_lora)

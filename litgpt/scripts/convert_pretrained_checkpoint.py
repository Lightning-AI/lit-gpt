# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

from pathlib import Path

import torch

from litgpt.utils import CLI, incremental_save, copy_config_files


@torch.inference_mode()
def convert_pretrained_checkpoint(checkpoint_dir: Path, output_dir: Path) -> None:
    """Convert a checkpoint after pretraining.

    The pretrained checkpoint contains optimizer states and several other metadata that are not needed after training
    is finished. This script will export the state-dict of the model and place it in the chosen output folder,
    which then can be loaded by other scripts for inference, evaluation, etc.

    Args:
        checkpoint_dir: Path to a checkpoint directory produced by ``litgpt.pretrain``.
        output_dir: The output folder where the converted state-dict file and config files will be saved to.
    """

    if output_dir.is_dir() and output_dir.glob("*"):
        raise FileExistsError(
            f"The output folder exists and is not empty: {str(output_dir)}."
            " Please delete it first or choose a different name."
        )

    output_dir.mkdir(parents=True)
    checkpoint_file = checkpoint_dir / "lit_model.pth"
    output_checkpoint_file = output_dir / "lit_model.pth"

    # TODO: Consolidate sharded checkpoint if applicable
    # Extract the model state dict and save to output folder
    with incremental_save(output_checkpoint_file) as saver:
        print("Processing", checkpoint_file)
        full_checkpoint = torch.load(str(checkpoint_file), mmap=True)
        loaded_state_dict = full_checkpoint["model"]
        converted_state_dict = {}
        for param_name, param in loaded_state_dict.items():
            saver.store_early(param)
            # remove prefix for compiled model (if any)
            param_name = param_name.replace("_orig_mod.", "")
            converted_state_dict[param_name] = param
        print(f"Saving converted checkpoint to {str(output_checkpoint_file)}.")
        saver.save(converted_state_dict)

    copy_config_files(checkpoint_dir, output_dir)


if __name__ == "__main__":
    CLI(convert_pretrained_checkpoint)

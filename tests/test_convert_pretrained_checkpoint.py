# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import os

import torch


def test_convert_pretrained_checkpoint(tmp_path, fake_checkpoint_dir):
    from litgpt.scripts.convert_pretrained_checkpoint import convert_pretrained_checkpoint

    # Pretend we made a checkpoint from pretraining
    pretrained_checkpoint = {
        "model": {"some.module.weight": torch.rand(2, 2), "_orig_mod.some.other.module.weight": torch.rand(2, 2)},
        "the_optimizer": "optimizer_state",
        "other": 1,
    }
    torch.save(pretrained_checkpoint, fake_checkpoint_dir / "lit_model.pth")

    convert_pretrained_checkpoint(checkpoint_dir=fake_checkpoint_dir, output_dir=(tmp_path / "converted"))

    assert set(os.listdir(tmp_path / "converted")) == {
        "lit_model.pth", "model_config.yaml", "tokenizer_config.json", "tokenizer.json"
    }
    converted_checkpoint = torch.load(tmp_path / "converted" / "lit_model.pth")
    assert list(converted_checkpoint.keys()) == ["some.module.weight", "some.other.module.weight"]

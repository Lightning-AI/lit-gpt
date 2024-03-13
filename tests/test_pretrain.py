# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import os
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock
from unittest.mock import Mock

import pytest
import torch
from conftest import RunIf
from torch.utils.data import DataLoader


@RunIf(min_cuda_gpus=2, standalone=True)
# Set CUDA_VISIBLE_DEVICES for FSDP hybrid-shard, if fewer GPUs are used than are available
@mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"})
# If we were to use `save_hyperparameters()`, we would have to patch `sys.argv` or otherwise
# the CLI would capture pytest args, but unfortunately patching would mess with subprocess
# launching, so we need to mock `save_hyperparameters()`
@mock.patch("litgpt.pretrain.save_hyperparameters")
def test_pretrain(_, tmp_path):
    from litgpt import pretrain
    from litgpt.args import EvalArgs, TrainArgs
    from litgpt.config import Config

    model_config = Config(block_size=2, n_layer=2, n_embd=8, n_head=4, padded_vocab_size=8)

    dataset = torch.tensor([[0, 1, 2], [3, 4, 5], [0, 1, 2]])
    dataloader = DataLoader(dataset)
    pretrain.get_dataloaders = Mock(return_value=(dataloader, dataloader))

    out_dir = tmp_path / "out"
    stdout = StringIO()
    with redirect_stdout(stdout):
        pretrain.setup(
            devices=2,
            model_config=model_config,
            out_dir=out_dir,
            train=TrainArgs(global_batch_size=2, max_tokens=16, save_interval=1, micro_batch_size=1, max_norm=1.0),
            eval=EvalArgs(interval=1, max_iters=1),
        )

    if torch.distributed.get_rank() == 0:
        # tmp_path is not the same across all ranks, run assert only on rank 0
        out_dir_contents = set(os.listdir(out_dir))
        checkpoint_dirs = {"step-00000001", "step-00000002", "step-00000003", "step-00000004"}
        assert checkpoint_dirs.issubset(out_dir_contents)
        assert all((out_dir / p).is_dir() for p in checkpoint_dirs)
        for checkpoint_dir in checkpoint_dirs:
            # the `tokenizer_dir` is None by default, so only 'lit_model.pth' shows here
            assert set(os.listdir(out_dir / checkpoint_dir)) == {"lit_model.pth", "model_config.yaml"}

        assert (out_dir / "logs" / "tensorboard" / "version_0").is_dir()

        # logs only appear on rank 0
        logs = stdout.getvalue()
        assert logs.count("(step)") == 4
        assert logs.count("val loss") == 4
        assert "Total parameters: 1,888" in logs

    torch.distributed.barrier()


def test_pretrain_model_name_and_config():
    from litgpt import pretrain
    from litgpt.config import Config

    with pytest.raises(ValueError, match="Only one of `model_name` or `model_config`"):
        pretrain.setup(model_name="tiny-llama-1.1b", model_config=Config(name="tiny-llama-1.1b"))


def test_init_out_dir(tmp_path):
    from litgpt.pretrain import init_out_dir

    relative_path = Path("./out")
    absolute_path = tmp_path / "out"
    assert init_out_dir(relative_path) == relative_path
    assert init_out_dir(absolute_path) == absolute_path

    with mock.patch.dict(os.environ, {"LIGHTNING_ARTIFACTS_DIR": "prefix"}):
        assert init_out_dir(relative_path) == Path("prefix") / relative_path
        assert init_out_dir(absolute_path) == absolute_path

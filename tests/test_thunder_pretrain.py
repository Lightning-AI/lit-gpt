import os
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock
from unittest.mock import Mock

import torch
from conftest import RunIf
from torch.utils.data import DataLoader

from litgpt import Config
from litgpt.args import EvalArgs, TrainArgs

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import extensions.thunder.pretrain as pretrain


@RunIf(min_cuda_gpus=1, thunder=True)
@mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"})
def test_pretrain(tmp_path, monkeypatch):
    model_config = Config(block_size=2, n_layer=2, n_embd=8, n_head=4, padded_vocab_size=8, attn_qkv_bias=True, attn_proj_bias=True, mlp_bias=True)

    dataset = torch.tensor([[0, 1, 2], [3, 4, 5], [0, 1, 2]])
    dataloader = DataLoader(dataset)
    monkeypatch.setattr(pretrain, "get_dataloaders", Mock(return_value=(dataloader, dataloader)))
    monkeypatch.setattr(pretrain, "save_hyperparameters", Mock())

    out_dir = tmp_path / "out"
    stdout = StringIO()
    with redirect_stdout(stdout):
        pretrain.setup(
            devices=1,
            model_config=model_config,
            out_dir=out_dir,
            train=TrainArgs(global_batch_size=2, max_tokens=16, save_interval=1, micro_batch_size=1, max_norm=1.0),
            eval=EvalArgs(interval=1, max_iters=1),
        )

    out_dir_contents = set(os.listdir(out_dir))
    checkpoint_dirs = {"step-00000001", "step-00000002", "step-00000003", "step-00000004"}
    assert checkpoint_dirs.issubset(out_dir_contents)
    assert all((out_dir / p).is_dir() for p in checkpoint_dirs)
    for checkpoint_dir in checkpoint_dirs:
        # the `tokenizer_dir` is None by default, so only 'lit_model.pth' shows here
        assert set(os.listdir(out_dir / checkpoint_dir)) == {"lit_model.pth", "model_config.yaml"}

    assert (out_dir / "logs" / "tensorboard" / "version_0").is_dir()

    logs = stdout.getvalue()
    assert logs.count("(step)") == 4
    assert logs.count("val loss") == 4
    assert "Total parameters: 1,888" in logs

# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import os
from contextlib import redirect_stdout
from dataclasses import asdict
from io import StringIO
from unittest import mock
from unittest.mock import Mock

import pytest
import torch
from conftest import RunIf
from lightning import Fabric
from lightning.fabric.plugins.precision.bitsandbytes import _BITSANDBYTES_AVAILABLE, BitsandbytesPrecision
from lightning.fabric.wrappers import _FabricOptimizer


def test_config_identical():
    import litgpt.adapter as gpt_adapter
    import litgpt.model as gpt

    name = "pythia-14m"
    base_config = asdict(gpt.Config.from_name(name))
    adapter_config = asdict(gpt_adapter.Config.from_name(name))
    del adapter_config["adapter_prompt_length"]
    del adapter_config["adapter_start_layer"]
    assert adapter_config == base_config

    with Fabric(accelerator="cpu").init_module(empty_init=True):
        base_model = gpt.GPT.from_name(name)
        adapter_model = gpt_adapter.GPT.from_name(name)
    assert adapter_model.lm_head.weight.shape == base_model.lm_head.weight.shape


def test_adapter_filter(tmp_path):
    from litgpt.adapter import GPT, adapter_filter

    fabric = Fabric(devices=1)
    model = GPT.from_name("pythia-14m", n_layer=4)
    save_path = tmp_path / "model.pth"
    fabric.save(save_path, {"model": model}, filter={"model": adapter_filter})
    saved = torch.load(save_path)["model"]

    expected = {
        "transformer.h.2.attn.adapter_wte.weight",
        "transformer.h.2.attn.gating_factor",
        "transformer.h.3.attn.adapter_wte.weight",
        "transformer.h.3.attn.gating_factor",
    }
    assert set(saved) == expected


@mock.patch.dict(os.environ, {"LT_ACCELERATOR": "cpu"})
def test_adapter_script(tmp_path, fake_checkpoint_dir, monkeypatch, alpaca_path):
    import litgpt.finetune.adapter as module
    from litgpt.data import Alpaca
    from litgpt.args import EvalArgs, TrainArgs
    from litgpt.config import name_to_config

    model_config = dict(block_size=128, n_layer=2, n_embd=8, n_head=4, padded_vocab_size=8, adapter_start_layer=0)
    monkeypatch.setitem(name_to_config, "tmp", model_config)

    monkeypatch.setattr(module, "load_checkpoint", Mock())

    tokenizer_mock = Mock()
    tokenizer_mock.return_value = tokenizer_mock
    tokenizer_mock.encode = lambda *_, **__: torch.tensor([3, 2, 1])
    monkeypatch.setattr(module, "Tokenizer", tokenizer_mock)

    out_dir = tmp_path / "out"
    stdout = StringIO()
    with redirect_stdout(stdout), mock.patch("sys.argv", ["adapter.py"]):
        module.setup(
            data=Alpaca(
                download_dir=alpaca_path.parent,
                file_name=alpaca_path.name,
                val_split_fraction=0.5,
                num_workers=0
            ),
            checkpoint_dir=fake_checkpoint_dir,
            out_dir=out_dir,
            precision="32-true",
            train=TrainArgs(global_batch_size=1, save_interval=2, epochs=1, max_steps=6, micro_batch_size=1),
            eval=EvalArgs(interval=2, max_iters=2, max_new_tokens=1),
        )

    out_dir_contents = set(os.listdir(out_dir))
    checkpoint_dirs = {"step-000002", "step-000004", "step-000006", "final"}
    assert checkpoint_dirs.issubset(out_dir_contents)
    assert all((out_dir / p).is_dir() for p in checkpoint_dirs)
    for checkpoint_dir in checkpoint_dirs:
        assert {p.name for p in (out_dir / checkpoint_dir).iterdir()} == {
            "lit_model.pth",
            "model_config.yaml",
            "tokenizer_config.json",
            "tokenizer.json",
            "hyperparameters.yaml",
            "prompt_style.yaml",
        }
    assert (out_dir / "version_0" / "metrics.csv").is_file()

    logs = stdout.getvalue()
    assert logs.count("(step)") == 6
    assert logs.count("val loss") == 3
    assert "of trainable parameters: 168" in logs


def test_adapter_gpt_init_weights():
    from litgpt.adapter import GPT, Config

    config = Config(n_layer=1, n_head=6, n_embd=12, block_size=1, vocab_size=1, adapter_start_layer=0)
    model = GPT(config)
    param = model.transformer.h[0].attn.gating_factor

    assert (param == 0).all()
    torch.nn.init.constant_(param, 1.23)
    assert (param != 0).any()
    model.apply(model._init_weights)
    assert (param == 0).all()


@RunIf(dynamo=True)
@torch.inference_mode()
def test_adapter_compile():
    from litgpt.adapter import GPT

    model = GPT.from_name("pythia-14m", n_layer=3)
    x = torch.randint(model.config.vocab_size, size=(2, model.config.block_size), dtype=torch.int64)

    from torch._dynamo.backends import debugging

    explanation = torch._dynamo.explain(model)(x)
    assert isinstance(explanation, debugging.ExplainOutput)
    assert explanation.graph_count == 1
    assert explanation.graph_break_count == 0

    model = GPT(model.config)
    model.set_kv_cache(2)
    input_pos = torch.arange(model.config.block_size)
    explanation = torch._dynamo.explain(model)(x, input_pos)
    assert isinstance(explanation, debugging.ExplainOutput)
    assert explanation.graph_count == 1
    assert explanation.graph_break_count == 0


@RunIf(min_cuda_gpus=1)
def test_adapter_bitsandbytes(monkeypatch, tmp_path, fake_checkpoint_dir, alpaca_path):
    from litgpt.config import name_to_config
    from litgpt.data import Alpaca
    import litgpt.finetune.adapter as module

    if not _BITSANDBYTES_AVAILABLE:
        pytest.skip("BNB not available")

    from bitsandbytes.optim import PagedAdamW

    model_config = dict(
        block_size=128, n_layer=2, n_embd=8, n_head=4, padded_vocab_size=8, adapter_start_layer=0, bias=True
    )
    monkeypatch.setitem(name_to_config, "tmp", model_config)

    tokenizer_mock = Mock()
    tokenizer_mock.return_value = tokenizer_mock
    tokenizer_mock.encode = lambda *_, **__: torch.tensor([3, 2, 1])
    monkeypatch.setattr(module, "Tokenizer", tokenizer_mock)

    monkeypatch.setattr(module, "load_checkpoint", Mock())
    train_mock = Mock()
    monkeypatch.setattr(module, "fit", train_mock)

    stdout = StringIO()
    with redirect_stdout(stdout), mock.patch("sys.argv", ["adapter.py"]):
        module.setup(
            data=Alpaca(
                download_dir=alpaca_path.parent,
                file_name=alpaca_path.name,
                val_split_fraction=0.5,
                num_workers=0,
            ),
            precision="16-true",
            quantize="bnb.nf4-dq",
            checkpoint_dir=fake_checkpoint_dir,
            out_dir=tmp_path,
        )

    args, kwargs = train_mock.call_args
    fabric, model, optimizer, *_ = args
    assert isinstance(fabric.strategy.precision, BitsandbytesPrecision)
    assert isinstance(optimizer, _FabricOptimizer)
    assert isinstance(optimizer._optimizer, PagedAdamW)

    dtype_to_name = {"torch.uint8": set(), "torch.float16": set()}
    for name, layer in model.named_parameters():
        name = name[len("_forward_module.") :]
        dtype_to_name[str(layer.dtype)].add(name)
    assert dtype_to_name == {
        "torch.float16": {
            "transformer.wte.weight",
            "transformer.h.0.norm_1.weight",
            "transformer.h.0.norm_1.bias",
            "transformer.h.0.attn.gating_factor",
            "transformer.h.0.attn.attn.bias",
            "transformer.h.0.attn.proj.bias",
            "transformer.h.0.attn.adapter_wte.weight",
            "transformer.h.0.norm_2.weight",
            "transformer.h.0.norm_2.bias",
            "transformer.h.0.mlp.fc.bias",
            "transformer.h.0.mlp.proj.bias",
            "transformer.h.1.norm_1.weight",
            "transformer.h.1.norm_1.bias",
            "transformer.h.1.attn.gating_factor",
            "transformer.h.1.attn.attn.bias",
            "transformer.h.1.attn.proj.bias",
            "transformer.h.1.attn.adapter_wte.weight",
            "transformer.h.1.norm_2.weight",
            "transformer.h.1.norm_2.bias",
            "transformer.h.1.mlp.fc.bias",
            "transformer.h.1.mlp.proj.bias",
            "transformer.ln_f.weight",
            "transformer.ln_f.bias",
        },
        "torch.uint8": {
            "lm_head.weight",
            "transformer.h.0.attn.attn.weight",
            "transformer.h.0.attn.proj.weight",
            "transformer.h.0.mlp.fc.weight",
            "transformer.h.0.mlp.proj.weight",
            "transformer.h.1.attn.attn.weight",
            "transformer.h.1.attn.proj.weight",
            "transformer.h.1.mlp.fc.weight",
            "transformer.h.1.mlp.proj.weight",
        },
    }

    assert {p.name for p in tmp_path.rglob("*.pth")} == {"lit_model.pth"}
    state_dict = torch.load(tmp_path / "final" / "lit_model.pth")
    assert len(state_dict) == 1
    dtype_to_name = {"torch.float16": set()}
    for name, layer in state_dict["model"].items():
        dtype_to_name[str(layer.dtype)].add(name)
    assert dtype_to_name == {
        "torch.float16": {
            "transformer.h.0.attn.adapter_wte.weight",
            "transformer.h.0.attn.gating_factor",
            "transformer.h.1.attn.adapter_wte.weight",
            "transformer.h.1.attn.gating_factor",
        }
    }

    logs = stdout.getvalue()
    assert "of trainable parameters: 168" in logs
    assert "of non trainable parameters: 1,888" in logs

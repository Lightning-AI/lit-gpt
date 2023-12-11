import sys
from contextlib import redirect_stdout
from io import StringIO
from itertools import product
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from conftest import RunIf
from lightning import Fabric

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import lit_gpt.config as config_module


def test_lora_layer_replacement():
    from lit_gpt.lora import GPT, Config, LoRALinear
    from lit_gpt.lora import CausalSelfAttention as LoRACausalSelfAttention

    config = Config(n_layer=2, n_head=4, n_embd=8, block_size=8, vocab_size=8, r=8, alpha=8, dropout=0.1)
    model = GPT(config)

    assert isinstance(model.transformer.h[0].attn, LoRACausalSelfAttention)
    assert isinstance(model.transformer.h[1].attn, LoRACausalSelfAttention)
    assert isinstance(model.lm_head, LoRALinear)
    assert isinstance(model.transformer.h[0].mlp.proj, LoRALinear)


def test_lora_merge():
    from lit_gpt.lora import GPT, Config, mark_only_lora_as_trainable, merge_lora_weights

    config = Config(
        n_layer=1,
        n_head=2,
        n_embd=8,
        block_size=8,
        vocab_size=8,
        r=8,
        alpha=8,
        dropout=0.1,
        to_query=True,
        to_value=True,
        to_projection=True,
    )
    model = GPT(config)
    model.train()

    initial_weight = model.transformer.h[0].attn.proj.linear.weight.clone()
    assert torch.equal(model.transformer.h[0].attn.proj.linear.weight, initial_weight)

    # perform an update to the LoRA weights
    mark_only_lora_as_trainable(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    y = model(torch.randint(0, 8, size=(2, 4), dtype=torch.int64))
    y.sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    # the weight remains unchanged (only lora A and B change)
    assert torch.equal(model.transformer.h[0].attn.proj.linear.weight, initial_weight)

    # calling merge() multiple times in a row should not merge multiple times
    merge_lora_weights(model)
    assert model.transformer.h[0].attn.attn.merged
    weight_after = model.transformer.h[0].attn.proj.linear.weight.clone()
    merge_lora_weights(model)
    merge_lora_weights(model)
    assert torch.equal(model.transformer.h[0].attn.proj.linear.weight, weight_after)

    # check that `W_after = W_initial + (A x B)`
    a = model.transformer.h[0].attn.proj.lora_A
    b = model.transformer.h[0].attn.proj.lora_B
    scaling = model.transformer.h[0].attn.proj.scaling
    delta_w = (b @ a) * scaling
    torch.testing.assert_close(weight_after, initial_weight + delta_w)


def test_lora_mqa_gqa():
    from lit_gpt.lora import GPT, Config

    # MHA
    config = Config(
        n_layer=1,
        n_head=4,
        n_embd=8,
        block_size=1,
        vocab_size=1,
        r=2,
        alpha=8,
        dropout=0.1,
        to_query=True,
        to_value=True,
    )
    assert config.n_query_groups == config.n_head
    model = GPT(config)
    attn = model.transformer.h[0].attn.attn
    assert attn.linear.weight.shape == (24, 8)
    assert attn.lora_A.shape == (4, 8)
    assert attn.lora_B.shape == (16, 2)
    assert torch.equal(attn.lora_ind, torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23]))
    x = torch.randint(0, 8, size=(3, 5, 16), dtype=torch.int64)
    assert attn.zero_pad(x).shape == (3, 5, 24)

    # MQA
    config.n_query_groups = 1
    model = GPT(config)
    attn = model.transformer.h[0].attn.attn
    assert attn.linear.weight.shape == (12, 8)
    assert attn.lora_A.shape == (4, 8)
    assert attn.lora_B.shape == (10, 2)
    assert torch.equal(attn.lora_ind, torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 10, 11]))
    x = torch.randint(0, 8, size=(3, 5, 10), dtype=torch.int64)
    assert attn.zero_pad(x).shape == (3, 5, 12)

    # GQA
    config.n_query_groups = 2
    model = GPT(config)
    attn = model.transformer.h[0].attn.attn
    assert attn.linear.weight.shape == (16, 8)
    assert attn.lora_A.shape == (4, 8)
    assert attn.lora_B.shape == (12, 2)
    assert torch.equal(attn.lora_ind, torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15]))
    x = torch.randint(0, 8, size=(3, 5, 12), dtype=torch.int64)
    assert attn.zero_pad(x).shape == (3, 5, 16)


def test_lora_filter(tmp_path):
    from lit_gpt.lora import GPT, lora_filter

    fabric = Fabric(devices=1)
    model = GPT.from_name("pythia-14m", n_layer=3, r=1, to_query=True, to_value=True)
    save_path = tmp_path / "model.pth"
    fabric.save(save_path, {"model": model}, filter={"model": lora_filter})
    saved = torch.load(save_path)["model"]

    expected = {
        "transformer.h.1.attn.attn.lora_B",
        "transformer.h.2.attn.attn.lora_B",
        "transformer.h.2.attn.attn.lora_A",
        "transformer.h.1.attn.attn.lora_A",
        "transformer.h.0.attn.attn.lora_A",
        "transformer.h.0.attn.attn.lora_B",
    }
    assert set(saved) == expected


def test_lora_script(tmp_path, fake_checkpoint_dir, monkeypatch):
    import finetune.lora as module

    module.gradient_accumulation_iters = 1
    module.save_interval = 2
    module.eval_interval = 2
    module.eval_iters = 2
    module.eval_max_new_tokens = 1
    module.max_iters = 6

    data = [
        {"input_ids": torch.tensor([0, 1, 2]), "labels": torch.tensor([1, 2, 3])},
        {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([2, 3, 4])},
    ]
    torch.save(data, tmp_path / "train.pt")
    torch.save(data, tmp_path / "test.pt")

    from lit_gpt.config import name_to_config

    model_config = dict(block_size=128, n_layer=2, n_embd=8, n_head=4, padded_vocab_size=8)
    monkeypatch.setitem(name_to_config, "tmp", model_config)
    monkeypatch.setattr(module, "load_checkpoint", Mock())

    tokenizer_mock = Mock()
    tokenizer_mock.return_value = tokenizer_mock
    tokenizer_mock.encode = lambda *_, **kwargs: torch.tensor([3, 2, 1], **kwargs)
    monkeypatch.setattr(module, "Tokenizer", tokenizer_mock)

    stdout = StringIO()
    with redirect_stdout(stdout):
        module.setup(data_dir=tmp_path, checkpoint_dir=fake_checkpoint_dir, out_dir=tmp_path, precision="32-true")

    assert {p.name for p in tmp_path.glob("*.pth")} == {
        "iter-000002-ckpt.pth",
        "iter-000004-ckpt.pth",
        "iter-000006-ckpt.pth",
        "lit_model_lora_finetuned.pth",
    }
    assert (tmp_path / "version_0" / "metrics.csv").is_file()

    logs = stdout.getvalue()
    assert logs.count("optimizer.step") == module.max_iters
    assert logs.count("val loss") == module.max_iters // module.eval_interval
    assert "of trainable parameters: 512" in logs


def test_lora_init_when_linear_overridden():
    from lit_gpt.lora import LoRAQKVLinear

    class MyLinear(torch.nn.Linear):
        def __init__(self, *args, **kwargs):
            # this needs to be implemented to demonstrate the failure
            super().__init__(*args, **kwargs)

    original_linear = torch.nn.Linear
    # Our bnb does this sort of monkey patching
    torch.nn.Linear = MyLinear
    layer = LoRAQKVLinear(1, 1, 1, 1)
    assert isinstance(layer.linear, original_linear)
    torch.nn.Linear = original_linear


@pytest.mark.parametrize(
    ("apply_to", "target_layer_names", "mlp_class_name"),
    (
        ("to_projection", "transformer.h.0.attn.proj", "GptNeoxMLP"),
        ("to_mlp", {"transformer.h.0.mlp.fc", "transformer.h.0.mlp.proj"}, "GptNeoxMLP"),
        ("to_head", "lm_head", "GptNeoxMLP"),
        ("to_projection", "transformer.h.0.attn.proj", "LLaMAMLP"),
        ("to_mlp", {"transformer.h.0.mlp.fc_1", "transformer.h.0.mlp.fc_2", "transformer.h.0.mlp.proj"}, "LLaMAMLP"),
        ("to_head", "lm_head", "LLaMAMLP"),
    ),
)
def test_lora_linear_utilization(apply_to, target_layer_names, mlp_class_name):
    from lit_gpt.lora import GPT, Config

    config = Config(
        n_layer=1,
        n_head=4,
        n_embd=8,
        block_size=1,
        vocab_size=1,
        r=2,
        alpha=8,
        dropout=0.1,
        _mlp_class=mlp_class_name,
        intermediate_size=8 * 3,
        **{apply_to: True},
    )
    model = GPT(config)
    state_dict = model.state_dict()

    if isinstance(target_layer_names, str):
        target_layer_names = {target_layer_names}
    lora_sublayers = (".lora_A", ".lora_B")

    # check that all the target layers have LoRA weights
    for layer_name in target_layer_names:
        for lora_sublayer in lora_sublayers:
            assert layer_name + lora_sublayer in state_dict

    # check that only target layers have LoRA weights
    lora_params = [k for k in state_dict if k.endswith(lora_sublayers)]
    lora_params = {k[:-7] for k in lora_params}
    assert lora_params == target_layer_names


@torch.inference_mode()
@pytest.mark.parametrize("apply_to", (None, "to_query", "to_key", "to_value", "to_projection", "to_mlp", "to_head"))
def test_lora_gpt_apply_lora_forward_no_exception(apply_to):
    from lit_gpt.lora import GPT, Config

    config = Config(n_layer=1, n_head=4, n_embd=8, block_size=1, vocab_size=1, r=2, alpha=8, dropout=0.1)
    if apply_to:
        setattr(config, apply_to, True)
    input_ids = torch.tensor([[1]])
    model = GPT(config)
    model.eval()

    model(input_ids)


@torch.inference_mode()
@pytest.mark.parametrize("n_query_groups", (1, 2, 3, 6))
@pytest.mark.parametrize("apply_to", product((False, True), repeat=3))
def test_lora_gpt_query_groups_merge_and_forward_no_exception(n_query_groups, apply_to):
    from lit_gpt.lora import GPT, Config, merge_lora_weights

    keys = ("to_query", "to_key", "to_value")
    values = apply_to
    apply_to = dict(zip(keys, values))

    config = Config(
        n_layer=1,
        n_head=6,
        n_embd=12,
        block_size=1,
        vocab_size=1,
        r=2,
        alpha=8,
        dropout=0.1,
        n_query_groups=n_query_groups,
        **apply_to,
    )
    model = GPT(config)
    merge_lora_weights(model)
    input_ids = torch.tensor([[1]])
    model(input_ids)


@torch.inference_mode()
@pytest.mark.parametrize("n_head", (1, 2, 3, 6, 12))
@pytest.mark.parametrize(
    "enable_lora",
    [
        (False, False, True),
        (False, True, False),
        (False, True, True),
        (True, False, False),
        (True, False, True),
        (True, True, False),
        (True, True, True),
    ],
)
def test_lora_qkv_linear_compare_conv1d(n_head, enable_lora):
    from torch.nn import functional as F

    from lit_gpt.lora import LoRAQKVLinear

    C = 12
    layer = LoRAQKVLinear(C, 3 * C, n_head=n_head, n_query_groups=n_head, r=2, enable_lora=enable_lora)
    x = torch.randn((1, 1, C))
    a = F.linear(x, layer.lora_A).transpose(-2, -1)  # after_A
    b = layer.lora_B.data.unsqueeze(-1)

    # original PyTorch conv1d function output
    conv1d_pytorch = F.conv1d(a, b, groups=sum(layer.enable_lora))

    # custom conv1d
    conv1d_custom = layer.conv1d(a, b)

    # custom conv1d forced to split, apply and concat tensors
    layer.n_head = layer.n_query_groups + 1
    conv1d_custom_forced = layer.conv1d(a, b)

    assert torch.allclose(conv1d_pytorch, conv1d_custom)
    assert torch.allclose(conv1d_pytorch, conv1d_custom_forced)


@pytest.mark.parametrize(("rank", "expected_merged"), ((0, False), (1, True)))
def test_lora_linear_weights_merged_status(rank, expected_merged):
    from lit_gpt.lora import LoRALinear

    layer = LoRALinear(10, 10, r=rank)
    assert not layer.merged
    layer.merge()
    assert layer.merged == expected_merged


@pytest.mark.parametrize(
    ("rank", "enable_lora", "expected_merged"),
    ((0, True, False), (1, True, True), (0, False, False), (1, False, False)),
)
def test_lora_qkv_linear_weights_merged_status(rank, enable_lora, expected_merged):
    from lit_gpt.lora import LoRAQKVLinear

    layer = LoRAQKVLinear(10, 3 * 10, n_head=2, n_query_groups=2, r=rank, enable_lora=enable_lora)
    assert not layer.merged
    layer.merge()
    assert layer.merged == expected_merged


@RunIf(min_cuda_gpus=1)
# platform dependent cuda issue: libbitsandbytes_cpu.so: undefined symbol: cquantize_blockwise_fp16_nf4
@pytest.mark.xfail(raises=AttributeError, strict=False)
# https://github.com/Lightning-AI/lit-gpt/issues/513
@pytest.mark.xfail(raises=RuntimeError, strict=True)
def test_lora_merge_with_quantize():
    from lightning.fabric.plugins.precision.bitsandbytes import _BITSANDBYTES_AVAILABLE, BitsandbytesPrecision

    if not _BITSANDBYTES_AVAILABLE:
        pytest.skip("BNB not available")

    from lit_gpt.lora import GPT, Config, mark_only_lora_as_trainable, merge_lora_weights

    config = Config(
        n_layer=1,
        n_head=2,
        n_embd=8,
        block_size=8,
        vocab_size=8,
        r=8,
        alpha=8,
        dropout=0.1,
        to_query=True,
        to_value=True,
        to_projection=True,
    )
    fabric = Fabric(devices=1, plugins=BitsandbytesPrecision("nf4", dtype=torch.bfloat16, ignore_modules={"lm_head"}))
    model = GPT(config)
    mark_only_lora_as_trainable(model)

    from bitsandbytes.optim import PagedAdamW

    optimizer = PagedAdamW(model.parameters(), lr=1.0)
    model, optimizer = fabric.setup(model, optimizer)

    model.train()

    attn_proj = model.transformer.h[0].attn.proj
    initial_weight = attn_proj.linear.weight.clone()

    # this was skipped
    assert model.lm_head.linear.weight.dtype is torch.float32
    assert attn_proj.linear.weight.dtype is torch.uint8

    # perform an update to the LoRA weights
    y = model(torch.randint(0, 8, size=(2, 4), dtype=torch.int64, device=fabric.device))
    loss = y.sum()
    fabric.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
    # the weight remains unchanged (only lora A and B change)
    assert torch.equal(attn_proj.linear.weight, initial_weight)

    # calling merge() multiple times in a row should not merge multiple times
    merge_lora_weights(model)
    assert attn_proj.merged
    weight_after = attn_proj.linear.weight.clone()
    merge_lora_weights(model)
    merge_lora_weights(model)
    assert torch.equal(attn_proj.linear.weight, weight_after)

    # check that `W_after = W_initial + (A x B)`
    delta_w = (attn_proj.lora_B @ attn_proj.lora_A) * attn_proj.scaling
    torch.testing.assert_close(weight_after, initial_weight + delta_w)


def test_lora_gpt_init_weights():
    from lit_gpt.lora import GPT, Config

    config = Config(n_layer=1, n_head=6, n_embd=12, block_size=1, vocab_size=1, r=2, alpha=8, to_head=True)
    model = GPT(config)
    param = model.lm_head.lora_B.data

    assert (param == 0).all()
    torch.nn.init.constant_(param, 1.23)
    assert (param != 0).any()
    model.apply(model._init_weights)
    assert (param == 0).all()


@pytest.mark.parametrize("name", [c["name"] for c in config_module.configs])
def test_base_model_can_be_lora_loaded(name):
    from lit_gpt.lora import GPT as LoRAGPT
    from lit_gpt.lora import lora_filter
    from lit_gpt.model import GPT as BaseGPT

    kwargs = {"n_layer": 2, "n_head": 8, "n_embd": 16, "padded_vocab_size": 32}
    base_model = BaseGPT.from_name(name, **kwargs)
    base_model_state_dict = base_model.state_dict()
    lora_model = LoRAGPT.from_name(
        name, **kwargs, r=1, to_query=True, to_key=True, to_value=True, to_projection=True, to_mlp=True, to_head=True
    )
    keys = lora_model.load_state_dict(base_model_state_dict, strict=False)
    assert not keys.unexpected_keys
    for k in keys.missing_keys:
        assert lora_filter(k, None)


@RunIf(dynamo=True)
@torch.inference_mode()
def test_lora_compile():
    from lit_gpt.lora import GPT

    model = GPT.from_name(
        "pythia-14m",
        n_layer=3,
        r=8,
        alpha=8,
        dropout=0.1,
        to_query=True,
        to_key=True,
        to_value=True,
        to_projection=True,
        to_mlp=True,
        to_head=True,
    )
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

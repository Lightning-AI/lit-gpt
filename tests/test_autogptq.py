from functools import reduce

import pytest
import torch
from conftest import RunIf


@RunIf(min_cuda_gpus=1)
@pytest.mark.parametrize("bits", [2, 3, 4, 8], ids=[f"{bit}bit" for bit in (2, 3, 4, 8)])
@pytest.mark.parametrize("group_size", [32, 128], ids=[f"{gs}gs" for gs in (32, 128)])
@pytest.mark.parametrize("mlp_class", ("GptNeoxMLP", "LLaMAMLP", "LLaMAMoE"))
def test_quantization(tmp_path, fake_checkpoint_dir, monkeypatch, bits, group_size, mlp_class):
    import json
    import math
    from contextlib import redirect_stdout
    from dataclasses import asdict
    from io import StringIO
    from pathlib import Path
    from unittest.mock import Mock

    from lit_gpt import GPT
    from lit_gpt.config import Config
    from quantize import autogptq as module

    # Prepare calibration data
    data = [
        {"input_ids": torch.tensor([0, 1, 2]), "labels": torch.tensor([1, 2, 3])},
        {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([2, 3, 4])},
    ]
    torch.save(data, tmp_path / "test.pt")

    # Prepare model's config
    config = Config(
        padded_vocab_size=10_000,
        n_layer=2,
        n_embd=128,
        n_head=8,
        n_query_groups=2,
        intermediate_size=256,
        _mlp_class=mlp_class,
        n_expert=4 if mlp_class == "LLaMAMoE" else 0,
        n_expert_per_token=2 if mlp_class == "LLaMAMoE" else 0,
    )
    config_path = fake_checkpoint_dir / "lit_config.json"
    config_path.write_text(json.dumps(asdict(config)))

    # Mock weights loading and a tokenizer
    monkeypatch.setattr(module, "load_checkpoint", Mock())
    tokenizer_mock = Mock()
    tokenizer_mock.eos_id.return_value = 0
    monkeypatch.setattr(module, "Tokenizer", tokenizer_mock)

    # Run quantization, check that the time for quantization is printed
    stdout = StringIO()
    with redirect_stdout(stdout):
        module.main(
            data_dir=tmp_path,
            checkpoint_dir=fake_checkpoint_dir,
            bits=bits,
            group_size=group_size,
        )
    assert "Quantization time" in stdout.getvalue()

    # Assert that the quantized model weights are saved
    quantized_model_dir = fake_checkpoint_dir / f"quantized/{bits}bit"
    files = [p.name for p in quantized_model_dir.glob("*")]
    assert "lit_model_gptq.pth" in files
    # Assert that the quantize config is saved
    assert "quantize_config.json" in files
    # Assert that the kernel type was saved
    assert "kernel" in json.loads(Path(quantized_model_dir / "quantize_config.json").read_text())

    # --- Validate the saved quantized weights ---
    quantized_state_dict = torch.load(quantized_model_dir / "lit_model_gptq.pth")

    # Create a reference model to check that the saved quantized weights have a proper shape
    reference_model = GPT(config)
    # Retrieve `inside_layer_modules` - they control what layers inside each Transformer Block are quantized
    autogptq = module.AutoGPTQ(reference_model, quantized=False, quantize_config=None)
    inside_layer_modules = autogptq.inside_layer_modules
    inside_layer_modules = sum(inside_layer_modules, [])

    for layer_name, weight in quantized_state_dict.items():
        reference_layer = reduce(getattr, layer_name.split(".")[:-1], reference_model)
        if not any(ilm in layer_name for ilm in inside_layer_modules):
            assert layer_name.endswith((".weight", ".bias"))
            assert weight.dtype != torch.int32
            assert weight.shape == reference_layer.weight.shape
        else:
            assert layer_name.endswith((".qweight", ".qzeros", ".scales", ".g_idx", ".bias"))
            if layer_name.endswith(".qweight"):
                assert weight.dtype == torch.int32
                assert weight.shape == (reference_layer.in_features // 32 * bits, reference_layer.out_features)
            elif layer_name.endswith(".qzeros"):
                assert weight.dtype == torch.int32
                assert weight.shape == (
                    math.ceil(reference_layer.in_features / group_size),
                    reference_layer.out_features // 32 * bits,
                )
            elif layer_name.endswith(".scales"):
                assert weight.dtype == torch.float16
                assert weight.shape == (
                    math.ceil(reference_layer.in_features / group_size),
                    reference_layer.out_features,
                )
            elif layer_name.endswith(".g_idx"):
                assert weight.dtype == torch.int32
                assert weight.shape == (reference_layer.in_features,)
            else:
                # bias
                assert weight.dtype == torch.float16
                assert weight.shape == reference_layer.bias.shape


@RunIf(min_cuda_gpus=1)
@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize("kernel", ("cuda_old", "cuda", "exllama", "exllamav2", "triton", "marlin"))
@pytest.mark.parametrize("bits", [2, 3, 4, 8], ids=[f"{bit}bit" for bit in (2, 3, 4, 8)])
@pytest.mark.parametrize("group_size", [32, 128], ids=[f"{gs}gs" for gs in (32, 128)])
@pytest.mark.parametrize("desc_act", (True, False), ids=["desc_act", ""])
@pytest.mark.parametrize("sym", (True, False), ids=["sym", ""])
@pytest.mark.parametrize("mlp_class", ("GptNeoxMLP", "LLaMAMLP", "LLaMAMoE"))
def test_layer_conversion(kernel, bits, group_size, desc_act, sym, mlp_class):
    import importlib
    from functools import reduce

    from lit_gpt import GPT
    from lit_gpt.config import Config
    from quantize.autogptq import AutoGPTQ, QuantizeConfig

    # Prepare model's config
    # NOTE: Marlin layer requires `in_features` to be divisible by 128
    # and `out_features` - by 126. Specify config accordingly
    config = Config(
        padded_vocab_size=10_000,
        n_layer=2,
        n_embd=256,
        n_head=8,
        n_query_groups=4,
        intermediate_size=256,
        _mlp_class=mlp_class,
        n_expert=4 if mlp_class == "LLaMAMoE" else 0,
        n_expert_per_token=2 if mlp_class == "LLaMAMoE" else 0,
    )

    # Create a model: it has to be on a GPU and with float16 precision
    device = "cuda:0"
    model = GPT(config).to(device=device, dtype=torch.float16)

    # Some kernels have specific requirements (e.g., precision). The code has to tell about it.
    # We should check it.
    if (
        (kernel == "triton" and bits == 3)
        or (kernel in ("exllama", "exllamav2", "marlin") and bits != 4)
        or (kernel == "marlin" and (group_size not in (-1, 128) or desc_act or not sym))
    ):
        with pytest.raises(ValueError, match="doesn't support") as e_info:
            quantize_config = QuantizeConfig(bits=bits, group_size=group_size, kernel=kernel)
        pytest.skip(str(e_info.value))

    quantize_config = QuantizeConfig(bits=bits, group_size=group_size, kernel=kernel, desc_act=desc_act, sym=sym)

    # Wrap the model in AutoGPTQ as it allows to convert "nn.Linear" layers to "QuantLinear"
    autogptq_model = AutoGPTQ(model, quantized=True, quantize_config=quantize_config)
    autogptq_model.convert_to_quantized(kernel, device)
    # Convert layers and run obligatory "post_init" method: initializes kernel's buffers
    autogptq_model.post_init()

    # Check that all the target layers were successfully converted
    inside_layer_modules = sum(autogptq_model.inside_layer_modules, [])

    QuantLinear = importlib.import_module(f"auto_gptq.nn_modules.qlinear.qlinear_{kernel}").QuantLinear

    for layer_name in model.state_dict():
        module = reduce(getattr, layer_name.split(".")[:-1], autogptq_model.model)
        if any(ilm in layer_name for ilm in inside_layer_modules):
            assert isinstance(module, QuantLinear)
            if kernel == "marlin":
                assert layer_name.endswith((".B", ".s", ".workspace", ".bias"))
            else:
                assert layer_name.endswith((".qweight", ".qzeros", ".scales", ".g_idx", ".bias"))
        else:
            assert not isinstance(module, QuantLinear)
            assert layer_name.endswith((".weight", ".bias"))

    # Run a forward pass, it should not fail
    x = torch.tensor([[9856, 23, 491, 1536, 304], [23, 345, 65, 123, 321]], dtype=torch.int32, device=device)
    model(x)


@RunIf(min_cuda_gpus=1)
@pytest.mark.parametrize("kernel", ("cuda_old", "cuda", "exllama", "exllamav2", "triton"))
def test_marlin_conversion(kernel, tmp_path):
    from functools import reduce

    from auto_gptq.nn_modules.qlinear.qlinear_marlin import QuantLinear

    from lit_gpt import GPT
    from lit_gpt.config import Config
    from quantize.autogptq import AutoGPTQ, QuantizeConfig

    # Prepare model's config
    # NOTE: Marlin layer requires `in_features` to be divisible by 128
    # and `out_features` - by 126. Specify config accordingly
    config = Config(padded_vocab_size=10_000, n_layer=2, n_embd=256, n_head=8, n_query_groups=4, intermediate_size=256)

    # Create a model: it has to be on a GPU and with float16 precision
    device = "cuda:0"
    model = GPT(config).to(device=device, dtype=torch.float16)

    quantize_config = QuantizeConfig(bits=4, group_size=128, desc_act=False, kernel=kernel)
    quantize_config.save_config(tmp_path / "quantize_config.json")

    # Wrap the model in AutoGPTQ as it allows to convert "nn.Linear" layers to "QuantLinear"
    autogptq_model = AutoGPTQ(model, quantized=True, quantize_config=quantize_config)
    autogptq_model.convert_to_quantized(kernel, device)
    # Convert layers and run obligatory "post_init" method: initializes kernel's buffers
    autogptq_model.post_init()

    # Convert to Marlin layers
    autogptq_model.convert_quantized_to_marlin(tmp_path)

    # Assert that all layers were converted
    inside_layer_modules = sum(autogptq_model.inside_layer_modules, [])

    for layer_name in model.state_dict():
        module = reduce(getattr, layer_name.split(".")[:-1], autogptq_model.model)
        if any(ilm in layer_name for ilm in inside_layer_modules):
            assert layer_name.endswith((".B", ".s", ".workspace", ".bias"))
            assert isinstance(module, QuantLinear)
        else:
            assert layer_name.endswith((".weight", ".bias"))
            assert not isinstance(module, QuantLinear)

    # Assert that the Marlin version of the model is cached
    assert "marlin_cache.pth" in [p.name for p in tmp_path.glob("*")]

    # Assert that the quantize config now knows that the Marlin was cached
    assert quantize_config.marlin_cached is True

    # Run a forward pass, it should not fail
    x = torch.tensor([[9856, 23, 491, 1536, 304], [23, 345, 65, 123, 321]], dtype=torch.int32, device=device)
    model(x)


@RunIf(min_cuda_gpus=1)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("kernel", ("cuda_old", "cuda", "exllama", "exllamav2", "triton"))
def test_strip_bias(bias, kernel):
    from lit_gpt import GPT
    from lit_gpt.config import Config
    from quantize.autogptq import AutoGPTQ, QuantizeConfig

    # Prepare model's config
    config = Config(
        padded_vocab_size=10_000, n_layer=2, n_embd=128, n_head=8, n_query_groups=4, intermediate_size=256, bias=bias
    )

    # Create a model: it has to be on a GPU and with float16 precision
    device = "cuda:0"
    model = GPT(config).to(device=device, dtype=torch.float16)

    # Wrap the model in AutoGPTQ as it allows to convert "nn.Linear" layers to "QuantLinear"
    quantize_config = QuantizeConfig(bits=4, group_size=128, desc_act=False)
    autogptq_model = AutoGPTQ(model, quantized=True, quantize_config=quantize_config)
    autogptq_model.convert_to_quantized(kernel, device)

    # Assert that bias is stripped if needed
    inside_layer_modules = tuple(sum(autogptq_model.inside_layer_modules, []))
    for name, module in model.named_modules():
        if name.endswith(inside_layer_modules):
            if bias:
                assert module.bias is not None
            else:
                assert module.bias is None

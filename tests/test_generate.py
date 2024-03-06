# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import json
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock
from unittest.mock import ANY, Mock, call

import pytest
import torch
from conftest import RunIf


@pytest.mark.parametrize(
    "max_seq_length", (pytest.param(10, marks=pytest.mark.xfail(raises=NotImplementedError, strict=True)), 20 + 5)
)
def test_generate(max_seq_length):
    import generate.base as generate
    from lit_gpt import GPT, Config

    T = 5
    input_idx = torch.randint(10, size=(T,))

    config = Config(block_size=128, vocab_size=16, n_layer=1, n_head=4, n_embd=8)
    model = GPT(config)
    model.max_seq_length = max_seq_length
    model.set_kv_cache(batch_size=1)
    max_new_tokens = 20

    multinomial_results = []

    def multinomial(*args, **kwargs):
        out = torch.multinomial(*args, **kwargs, num_samples=1)
        multinomial_results.append(out)
        return out

    with mock.patch("generate.base.multinomial_num_samples_1", multinomial):
        out = generate.generate(model, input_idx, T + max_new_tokens, top_k=4)

    assert out.size(0) == T + max_new_tokens
    multinomial_results = torch.hstack(multinomial_results)
    expected = torch.cat((input_idx, multinomial_results))
    assert out.shape == expected.shape
    torch.testing.assert_close(out, expected)


def test_main(fake_checkpoint_dir, monkeypatch, tensor_like):
    import generate.base as generate

    config_path = fake_checkpoint_dir / "lit_config.json"
    config = {"block_size": 128, "vocab_size": 50, "n_layer": 2, "n_head": 4, "n_embd": 8, "rotary_percentage": 1}
    config_path.write_text(json.dumps(config))

    module_mock = Mock()
    module_mock.config.block_size = 128
    load_mock = Mock()
    load_mock.return_value = load_mock
    monkeypatch.setattr(generate, "load_checkpoint", load_mock)
    tokenizer_mock = Mock()
    tokenizer_mock.return_value.encode.return_value = torch.tensor([1, 2, 3])
    tokenizer_mock.return_value.decode.return_value = "foo bar baz"
    monkeypatch.setattr(generate, "Tokenizer", tokenizer_mock)
    generate_mock = Mock()
    generate_mock.return_value = torch.tensor([3, 2, 1])
    monkeypatch.setattr(generate, "generate", generate_mock)

    num_samples = 2
    out, err = StringIO(), StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        generate.main(temperature=2.0, top_k=2, num_samples=num_samples, checkpoint_dir=fake_checkpoint_dir)

    assert len(tokenizer_mock.return_value.decode.mock_calls) == num_samples
    assert torch.allclose(tokenizer_mock.return_value.decode.call_args[0][0], generate_mock.return_value)
    assert (
        generate_mock.mock_calls
        == [call(ANY, tensor_like, 53, temperature=2.0, top_k=2, eos_id=tokenizer_mock.return_value.eos_id)]
        * num_samples
    )
    # only the generated result is printed to stdout
    assert out.getvalue() == "foo bar baz\n" * num_samples

    assert "'padded_vocab_size': 512, 'n_layer': 2, 'n_head': 4" in err.getvalue()


@RunIf(min_cuda_gpus=1)
@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize("bits", [2, 3, 4, 8], ids=[f"{bit}bit" for bit in (2, 3, 4, 8)])
@pytest.mark.parametrize("kernel", (None, "cuda_old", "cuda", "triton", "exllama", "exllamav2", "marlin"))
def test_main_with_gptq(tmp_path, fake_checkpoint_dir, monkeypatch, tensor_like, bits, kernel):
    if (kernel == "triton" and bits == 3) or (kernel in ("exllama", "exllamav2", "marlin") and bits != 4):
        pytest.skip(f"{kernel} doesn't support {bits}bit precision.")

    import json
    from contextlib import redirect_stdout
    from dataclasses import asdict
    from io import StringIO
    from pathlib import Path
    from unittest.mock import Mock

    import torch

    import generate.base as generate
    from lit_gpt.config import Config
    from quantize import autogptq as module

    # ------------------- Quantization -------------------
    # Instead of trying to correctly mock quantized weights and to not shoot in the leg along the process
    # it's better to do a quantization and use the quantized weights with a corresponding quantize
    # config during inference

    # Prepare calibration data
    data = [
        {"input_ids": torch.tensor([0, 1, 2]), "labels": torch.tensor([1, 2, 3])},
        {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([2, 3, 4])},
    ]
    torch.save(data, tmp_path / "test.pt")

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
            group_size=128,
            desc_act=False,  # to make it compatible with Marlin kernel
            static_groups=False,
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

    # ------------------- Inference -------------------

    tokenizer_mock = Mock()
    tokenizer_mock.return_value.encode.return_value = torch.tensor([1, 2, 3], device="cuda")
    tokenizer_mock.return_value.decode.return_value = "foo bar baz"
    monkeypatch.setattr(generate, "Tokenizer", tokenizer_mock)
    generate_mock = Mock()
    generate_mock.return_value = torch.tensor([3, 2, 1])
    monkeypatch.setattr(generate, "generate", generate_mock)

    num_samples = 2
    stdout, stderr = StringIO(), StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        generate.main(
            temperature=2.0,
            top_k=2,
            num_samples=num_samples,
            checkpoint_dir=fake_checkpoint_dir,
            quantize=f"gptq.int{bits}",
            gptq_kernel=kernel,
        )
    assert len(tokenizer_mock.return_value.decode.mock_calls) == num_samples
    assert torch.allclose(tokenizer_mock.return_value.decode.call_args[0][0], generate_mock.return_value)
    assert (
        generate_mock.mock_calls
        == [call(ANY, tensor_like, 53, temperature=2.0, top_k=2, eos_id=tokenizer_mock.return_value.eos_id)]
        * num_samples
    )
    # after system info the generated output is printed
    assert stdout.getvalue().endswith("foo bar baz\n" * num_samples)

    assert "'padded_vocab_size': 10000, 'n_layer': 2, 'n_head': 8" in stderr.getvalue()


def test_cli():
    cli_path = Path(__file__).parent.parent / "generate" / "base.py"
    output = subprocess.check_output([sys.executable, cli_path, "-h"])
    output = str(output.decode())
    assert "Generates text samples" in output


@pytest.mark.parametrize("temperature", (0.0, 1.0, 0.5))
def test_sample(temperature):
    from generate.base import sample

    # shape: 2x3x5
    logits = torch.tensor(
        [
            [[24, 4, 98, 77, 47], [65, 70, 32, 67, 24], [92, 32, 88, 36, 62]],
            [[85, 79, 57, 68, 50], [89, 46, 72, 45, 32], [68, 96, 68, 24, 36]],
        ]
    )
    token = sample(logits, temperature=temperature)

    assert token.shape == (1,)
    # sample is batch size 1 only for now - this should be [0, 1] once batched generation is supported
    assert token.tolist() == [0]

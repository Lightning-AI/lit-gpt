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
import yaml


@pytest.mark.parametrize(
    "max_seq_length", (pytest.param(10, marks=pytest.mark.xfail(raises=NotImplementedError, strict=True)), 20 + 5)
)
def test_generate(max_seq_length):
    import litgpt.generate.base as generate
    from litgpt import GPT, Config

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

    with mock.patch("litgpt.generate.base.multinomial_num_samples_1", multinomial):
        out = generate.generate(model, input_idx, T + max_new_tokens, top_k=4)

    assert out.size(0) == T + max_new_tokens
    multinomial_results = torch.hstack(multinomial_results)
    expected = torch.cat((input_idx, multinomial_results))
    assert out.shape == expected.shape
    torch.testing.assert_close(out, expected)


def test_main(fake_checkpoint_dir, monkeypatch, tensor_like):
    import litgpt.generate.base as generate

    config_path = fake_checkpoint_dir / "model_config.yaml"
    config = {"block_size": 128, "vocab_size": 50, "n_layer": 2, "n_head": 4, "n_embd": 8, "rotary_percentage": 1}
    config_path.write_text(yaml.dump(config))

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


@pytest.mark.parametrize("mode", ["file", "entrypoint"])
def test_cli(mode):
    if mode == "file":
        cli_path = Path(__file__).parent.parent / "litgpt/generate/base.py"
        args = [sys.executable, cli_path, "-h"]
    else:
        args = ["litgpt", "generate", "base", "-h"]
    output = subprocess.check_output(args)
    output = str(output.decode())
    assert "Generates text samples" in output


@pytest.mark.parametrize("temperature", (0.0, 1.0, 0.5))
def test_sample(temperature):
    from litgpt.generate.base import sample

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

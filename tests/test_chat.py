# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from itertools import repeat
from pathlib import Path
from unittest.mock import ANY, MagicMock, Mock, call, patch

import pytest
import torch
import yaml


@pytest.mark.parametrize(
    ("generated", "stop_tokens", "expected"),
    [
        (repeat(1), (), [1] * 8),
        ([1, 2, 3, 0], ([0],), [1, 2, 3]),
        ([1, 2, 3, 0], ([9], [2, 4], [1, 2, 3, 0]), []),
        ([1, 2, 3, 0, 0], ([0, 0, 0], [0, 0]), [1, 2, 3]),
        ([3, 1, 2], ([1, 2], [3]), []),
        ([1, 2, 3, 0, 3, 2, 1, 0], ([4, 3, 2, 1], [2, 4]), [1, 2, 3, 0, 3, 2, 1, 0]),
    ],
)
def test_generate(monkeypatch, generated, stop_tokens, expected):
    import litgpt.chat.base as chat
    import litgpt.generate.base as generate

    input_idx = torch.tensor([5, 3])
    max_returned_tokens = len(input_idx) + 8
    model = MagicMock()
    model.config.block_size = 100
    model.max_seq_length = 100
    it = iter(generated)

    def multinomial(*_, **__):
        out = next(it)
        return torch.tensor([out])

    monkeypatch.setattr(generate, "multinomial_num_samples_1", multinomial)
    actual = chat.generate(model, input_idx, max_returned_tokens, stop_tokens=stop_tokens)
    actual = list(actual)

    assert len(actual) == len(expected)
    if not actual:
        assert actual == expected
    else:
        for t in actual:
            assert t.dtype == torch.long
        assert torch.cat(actual).tolist() == expected


@pytest.mark.parametrize("tokenizer_backend", ["huggingface", "sentencepiece"])
def test_decode(tokenizer_backend):
    from lightning.fabric import Fabric

    import litgpt.chat.base as chat

    class Tokenizer:
        backend = tokenizer_backend
        id2token = {1: "foo ", 2: "bar ", 3: "baz "}

        def decode(self, tensor: torch.Tensor) -> str:
            tensor = [tensor] if tensor.ndim == 0 else tensor
            return "".join(self.id2token[int(value)] for value in tensor)

    tokenizer_mock = Tokenizer()

    fabric = Fabric(devices=1, accelerator="cpu")

    token_stream = torch.tensor([3, 2, 1])
    out, err = StringIO(), StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        chat.decode(fabric, tokenizer_mock, token_stream)

    assert out.getvalue() == "baz bar foo "


@patch("litgpt.chat.base.input")
@pytest.mark.parametrize("stop_iteration", [KeyboardInterrupt, ""])
def test_main(mocked_input, stop_iteration, fake_checkpoint_dir, monkeypatch, tensor_like):
    import litgpt.chat.base as chat

    # these values will be iteratively provided for each `input()` call
    mocked_input.side_effect = ["Hello", stop_iteration]

    config_path = fake_checkpoint_dir / "model_config.yaml"
    config = {"block_size": 128, "vocab_size": 50, "n_layer": 2, "n_head": 4, "n_embd": 8, "rotary_percentage": 1}
    config_path.write_text(yaml.dump(config))

    load_mock = Mock()
    load_mock.return_value = load_mock
    monkeypatch.setattr(chat, "load_checkpoint", load_mock)
    tokenizer_mock = Mock()
    tokenizer_mock.return_value.backend = "sentencepiece"
    tokenizer_mock.return_value.encode.return_value = torch.tensor([1, 2, 3])
    tokenizer_mock.return_value.decode.return_value = "foo bar baz"
    monkeypatch.setattr(chat, "Tokenizer", tokenizer_mock)
    generate_mock = Mock()
    generate_mock.return_value = torch.tensor([3, 2, 1])
    monkeypatch.setattr(chat, "generate", generate_mock)

    out, err = StringIO(), StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        chat.main(temperature=2.0, top_k=2, checkpoint_dir=fake_checkpoint_dir)

    # decoding is done per each generated item
    assert len(tokenizer_mock.return_value.decode.mock_calls) == generate_mock.return_value.numel()
    assert torch.allclose(tokenizer_mock.return_value.decode.call_args[0][0], generate_mock.return_value)
    assert generate_mock.mock_calls == [
        call(ANY, tensor_like, 128, temperature=2.0, top_k=2, stop_tokens=([tokenizer_mock.return_value.eos_id],))
    ]
    # # only the generated result is printed to stdout
    assert out.getvalue() == ">> Reply: foo bar baz\n"

    assert "'padded_vocab_size': 512, 'n_layer': 2, 'n_head': 4" in err.getvalue()


@pytest.mark.parametrize("mode", ["file", "entrypoint"])
def test_cli(mode):
    if mode == "file":
        cli_path = Path(__file__).parent.parent / "litgpt/chat/base.py"
        args = [sys.executable, cli_path, "-h"]
    else:
        args = ["litgpt", "chat", "-h"]
    output = subprocess.check_output(args)
    output = str(output.decode())
    assert "Starts a conversation" in output

import importlib
import importlib.util
from pathlib import Path
from unittest import mock
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize(["script_file", "config_file"], [
    ("litgpt/pretrain.py", "pretrain/debug.yaml"),
    ("litgpt/pretrain.py", "pretrain/tinyllama.yaml"),
    ("litgpt/pretrain.py", "pretrain/tinystories.yaml"),
    ("litgpt/pretrain.py", "https://raw.githubusercontent.com/Lightning-AI/litgpt/wip/config_hub/pretrain/tinystories.yaml"),
    ("litgpt/finetune/full.py", "finetune/llama-2-7b/full.yaml"),
    ("litgpt/finetune/lora.py", "finetune/llama-2-7b/lora.yaml"),
    ("litgpt/finetune/lora.py", "finetune/tiny-llama/lora.yaml"),
])
def test_config_help(script_file, config_file, monkeypatch, tmp_path):
    """Test that configs validate against the signature in the scripts."""
    from litgpt.utils import CLI

    script_file = Path(__file__).parent.parent / script_file
    assert script_file.is_file()
    if "http" not in str(config_file):
        config_file = Path(__file__).parent.parent / "config_hub" / config_file
        assert config_file.is_file()

    spec = importlib.util.spec_from_file_location(str(script_file.parent.name), script_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module.main = Mock()
    module.Tokenizer = Mock()

    with mock.patch("sys.argv", [script_file.name, "--config", str(config_file), "--devices", "1"]):
        CLI(module.setup)

    module.main.assert_called_once()

# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import json
import sys
from pathlib import Path

import pytest
import yaml

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import litgpt.config as config_module


def test_config():
    from litgpt import Config

    config = Config()
    assert config.name == ""
    assert config.block_size == 4096

    config = Config(block_size=2048)
    assert config.block_size == 2048

    config = Config.from_name("pythia-14m")
    assert config.block_size == 512

    config = Config.from_name("pythia-14m", block_size=4096)
    assert config.block_size == 4096

    config = Config(hf_config={"name": "pythia-14m"})
    assert config.name == "pythia-14m"


def test_from_hf_name():
    from litgpt import Config

    # by short-hand name
    config0 = Config.from_name("tiny-llama-1.1b")
    # or by huggingface hub repo name
    config1 = Config.from_name("TinyLlama-1.1B-intermediate-step-1431k-3T")
    assert config0 == config1


@pytest.mark.parametrize("config", config_module.configs, ids=[c["name"] for c in config_module.configs])
def test_short_and_hf_names_are_equal_unless_on_purpose(config):
    from litgpt import Config

    # by short-hand name
    config0 = Config.from_name(config["name"])
    # or by huggingface hub repo name
    config1 = Config.from_name(config["hf_config"]["name"])
    assert config0.name == config1.name


def test_nonexisting_name():
    from litgpt import Config

    with pytest.raises(ValueError, match="not a supported"):
        Config.from_name("foobar")


def test_from_checkpoint(tmp_path):
    from litgpt import Config

    # 1. Neither `lit_config.py` nor matching config exists.
    with pytest.raises(FileNotFoundError, match="neither 'model_config.yaml' nor matching config exists"):
        Config.from_checkpoint(tmp_path / "non_existing_checkpoint")

    # 2. If `lit_config.py` doesn't exists, but there is a matching config in `litgpt/config.py`.
    config = Config.from_checkpoint(tmp_path / "pythia-14m")
    assert config.name == "pythia-14m"
    assert config.block_size == 512
    assert config.n_layer == 6

    # 3. If only `lit_config.py` exists.
    config_data = {"name": "pythia-14m", "block_size": 24, "n_layer": 2}
    with open(tmp_path / "model_config.yaml", "w") as file:
        yaml.dump(config_data, file)
    config = Config.from_checkpoint(tmp_path)
    assert config.name == "pythia-14m"
    assert config.block_size == 24
    assert config.n_layer == 2

    # 4. Both `lit_config.py` and a matching config exist, but `lit_config.py` supersedes matching config
    (tmp_path / "pythia-14m").mkdir()
    with open(tmp_path / "pythia-14m/model_config.yaml", "w") as file:
        yaml.dump(config_data, file)
    config = Config.from_checkpoint(tmp_path / "pythia-14m")
    assert config.name == "pythia-14m"
    assert config.block_size == 24
    assert config.n_layer == 2


@pytest.mark.parametrize("head_size", [None, 128])
def test_head_size(head_size):
    from litgpt import Config

    config = Config(head_size)

    assert config.head_size == head_size or config.n_embd // config.n_head

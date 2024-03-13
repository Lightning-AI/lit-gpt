# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import os
import shutil
import sys
from pathlib import Path
from typing import List

import pytest
import torch
from lightning.fabric.utilities.testing import _runif_reasons

wd = Path(__file__).parent.parent.absolute()


@pytest.fixture(autouse=True)
def add_wd_to_path():
    # this adds support for running tests without the package installed
    sys.path.append(str(wd))


@pytest.fixture()
def fake_checkpoint_dir(tmp_path):
    os.chdir(tmp_path)
    checkpoint_dir = tmp_path / "checkpoints" / "tmp"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "lit_model.pth").touch()
    (checkpoint_dir / "model_config.yaml").touch()
    (checkpoint_dir / "tokenizer.json").touch()
    (checkpoint_dir / "tokenizer_config.json").touch()
    return checkpoint_dir


class TensorLike:
    def __eq__(self, other):
        return isinstance(other, torch.Tensor)


@pytest.fixture()
def tensor_like():
    return TensorLike()


class FloatLike:
    def __eq__(self, other):
        return not isinstance(other, int) and isinstance(other, float)


@pytest.fixture()
def float_like():
    return FloatLike()


@pytest.fixture(autouse=True)
def restore_default_dtype():
    # just in case
    torch.set_default_dtype(torch.float32)


class MockTokenizer:
    """A dummy tokenizer that encodes each character as its ASCII code."""
    eos_id = 1

    def encode(self, text: str, eos: bool = False, max_length: int = -1) -> torch.Tensor:
        output = [ord(c) for c in text]
        if eos:
            output.append(self.eos_id)
        output = output[:max_length] if max_length > 0 else output
        return torch.tensor(output)

    def decode(self, tokens: torch.Tensor) -> str:
        return "".join(chr(int(t)) for t in tokens.tolist())


@pytest.fixture()
def mock_tokenizer():
    return MockTokenizer()


@pytest.fixture()
def alpaca_path(tmp_path):
    file = Path(__file__).parent / "data" / "fixtures" / "alpaca.json"
    shutil.copyfile(file, tmp_path / "alpaca.json")
    return tmp_path / "alpaca.json"


@pytest.fixture()
def dolly_path(tmp_path):
    file = Path(__file__).parent / "data" / "fixtures" / "dolly.json"
    shutil.copyfile(file, tmp_path / "dolly.json")
    return tmp_path / "dolly.json"


@pytest.fixture()
def longform_path(tmp_path):
    path = tmp_path / "longform"
    path.mkdir()
    for split in ("train", "val"):
        file = Path(__file__).parent / "data" / "fixtures" / f"longform_{split}.json"
        shutil.copyfile(file, path / f"{split}.json")
    return path


def RunIf(**kwargs):
    reasons, marker_kwargs = _runif_reasons(**kwargs)
    return pytest.mark.skipif(condition=len(reasons) > 0, reason=f"Requires: [{' + '.join(reasons)}]", **marker_kwargs)


# https://github.com/Lightning-AI/lightning/blob/6e517bd55b50166138ce6ab915abd4547702994b/tests/tests_fabric/conftest.py#L140
def pytest_collection_modifyitems(items: List[pytest.Function], config: pytest.Config) -> None:
    initial_size = len(items)
    conditions = []
    filtered, skipped = 0, 0

    options = {"standalone": "PL_RUN_STANDALONE_TESTS", "min_cuda_gpus": "PL_RUN_CUDA_TESTS"}
    if os.getenv(options["standalone"], "0") == "1" and os.getenv(options["min_cuda_gpus"], "0") == "1":
        # special case: we don't have a CPU job for standalone tests, so we shouldn't run only cuda tests.
        # by deleting the key, we avoid filtering out the CPU tests
        del options["min_cuda_gpus"]

    for kwarg, env_var in options.items():
        # this will compute the intersection of all tests selected per environment variable
        if os.getenv(env_var, "0") == "1":
            conditions.append(env_var)
            for i, test in reversed(list(enumerate(items))):  # loop in reverse, since we are going to pop items
                already_skipped = any(marker.name == "skip" for marker in test.own_markers)
                if already_skipped:
                    # the test was going to be skipped anyway, filter it out
                    items.pop(i)
                    skipped += 1
                    continue
                has_runif_with_kwarg = any(
                    marker.name == "skipif" and marker.kwargs.get(kwarg) for marker in test.own_markers
                )
                if not has_runif_with_kwarg:
                    # the test has `@RunIf(kwarg=True)`, filter it out
                    items.pop(i)
                    filtered += 1

    if config.option.verbose >= 0 and (filtered or skipped):
        writer = config.get_terminal_writer()
        writer.write(
            f"\nThe number of tests has been filtered from {initial_size} to {initial_size - filtered} after the"
            f" filters {conditions}.\n{skipped} tests are marked as unconditional skips.\nIn total,"
            f" {len(items)} tests will run.\n",
            flush=True,
            bold=True,
            purple=True,  # oh yeah, branded pytest messages
        )

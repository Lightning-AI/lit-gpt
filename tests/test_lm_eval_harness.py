# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import subprocess
import sys
from pathlib import Path
from unittest.mock import ANY, Mock

import datasets
import pytest
import yaml
from lightning import Fabric


@pytest.mark.xfail(
    raises=(datasets.builder.DatasetGenerationError, NotImplementedError),
    strict=False,
    match="Loading a dataset cached in a LocalFileSystem is not supported",
)
def test_run_eval(tmp_path, float_like):
    from eval.lm_eval_harness import EvalHarnessBase
    from litgpt.model import GPT
    from litgpt.tokenizer import Tokenizer
    from litgpt.scripts.download import download_from_hub

    fabric = Fabric(devices=1)
    with fabric.init_module():
        model = GPT.from_name("pythia-14m")
    download_from_hub(repo_id="EleutherAI/pythia-14m", tokenizer_only=True, checkpoint_dir=tmp_path)
    tokenizer = Tokenizer(tmp_path / "EleutherAI/pythia-14m")

    eval_harness = EvalHarnessBase(fabric, model, tokenizer, 1)
    results = eval_harness.run_eval(
        eval_tasks=["truthfulqa_mc", "hellaswag", "coqa"], limit=2, bootstrap_iters=2, num_fewshot=0, no_cache=True
    )
    assert results == {
        "config": {
            "batch_size": 1,
            "bootstrap_iters": 2,
            "device": ANY,
            "limit": 2,
            "model": "pythia-14m",
            "no_cache": True,
            "num_fewshot": 0,
        },
        "results": {
            "hellaswag": {
                "acc": float_like,
                "acc_norm": float_like,
                "acc_norm_stderr": float_like,
                "acc_stderr": float_like,
            },
            "coqa": {"f1": float_like, "f1_stderr": float_like, "em": float_like, "em_stderr": float_like},
            "truthfulqa_mc": {"mc1": float_like, "mc1_stderr": float_like, "mc2": float_like, "mc2_stderr": float_like},
        },
        "versions": {"hellaswag": 0, "coqa": 1, "truthfulqa_mc": 1},
    }


def test_eval_script(tmp_path, fake_checkpoint_dir, monkeypatch):
    import eval.lm_eval_harness as module

    model_config = dict(block_size=128, n_layer=2, n_embd=8, n_head=4, padded_vocab_size=8)
    with open(fake_checkpoint_dir / "model_config.yaml", "w") as fp:
        yaml.dump(model_config, fp)
    monkeypatch.setattr(module, "load_checkpoint", Mock())

    tokenizer_mock = Mock()
    monkeypatch.setattr(module, "Tokenizer", tokenizer_mock)

    run_eval_mock = Mock()
    run_eval_mock.return_value = {"foo": "test"}
    monkeypatch.setattr(module.EvalHarnessBase, "run_eval", run_eval_mock)

    output_folder = tmp_path / "output"
    assert not output_folder.exists()

    module.run_eval_harness(
        checkpoint_dir=fake_checkpoint_dir, precision="32-true", save_filepath=(output_folder / "results.json")
    )

    run_eval_mock.assert_called_once_with(
        ["arc_challenge", "piqa", "hellaswag", "hendrycksTest-*"], 0, None, 100000, True
    )
    assert (output_folder / "results.json").read_text() == '{"foo": "test"}'


def test_cli():
    cli_path = Path(__file__).parent.parent / "eval" / "lm_eval_harness.py"
    output = subprocess.check_output([sys.executable, cli_path, "-h"])
    output = str(output.decode())
    assert "run_eval_harness" in output

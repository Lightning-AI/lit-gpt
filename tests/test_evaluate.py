# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import sys
import os
from dataclasses import asdict
from pathlib import Path
from unittest import mock
import litgpt.eval.evaluate as module
from contextlib import redirect_stdout
from io import StringIO
import subprocess

import datasets
import pytest
import yaml
import torch


from litgpt import GPT, Config

from litgpt.scripts.download import download_from_hub
from litgpt.eval.evaluate import safe_safetensors, prepare_results
from litgpt.scripts.convert_lit_checkpoint import convert_lit_checkpoint
from lm_eval import evaluator

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


@pytest.mark.xfail(
    raises=(datasets.builder.DatasetGenerationError, NotImplementedError),
    strict=False,
    match="Loading a dataset cached in a LocalFileSystem is not supported",
)
@mock.patch.dict(os.environ, {"LT_ACCELERATOR": "cpu"})
def test_evaluate_script(tmp_path, fake_checkpoint_dir, monkeypatch):
    ours_config = Config.from_name("pythia-14m")
    ours_model = GPT(ours_config)
    checkpoint_path = fake_checkpoint_dir / "lit_model.pth"
    config_path = fake_checkpoint_dir / "model_config.yaml"
    torch.save(ours_model.state_dict(), checkpoint_path)
    with open(config_path, "w") as fp:
        yaml.dump(asdict(ours_config), fp)
    output_dir = tmp_path / "out_dir"

    fn_kwargs = dict(
        checkpoint_dir=fake_checkpoint_dir,
        out_dir=output_dir,
    )
    stdout = StringIO()
    with redirect_stdout(stdout), mock.patch("sys.argv", ["evaluate.py"]):
        module.convert_and_evaluate(**fn_kwargs)


def test_cli():
    cli_path = Path(__file__).parent.parent / "eval" / "evaluate.py"
    output = subprocess.check_output([sys.executable, cli_path, "-h"])
    output = str(output.decode())
    assert "evaluate" in output

"""
def test_run_eval(tmp_path, float_like):
    repo_id = "EleutherAI/pythia-14m"
    download_from_hub(repo_id=repo_id, checkpoint_dir=tmp_path)

    checkpoint_path = Path(tmp_path) / Path(repo_id)

    convert_lit_checkpoint(checkpoint_dir=checkpoint_path, output_dir=checkpoint_path)
    safe_safetensors(out_dir=checkpoint_path, repo_id=repo_id)

    eval_tasks = "coqa,hellaswag"
    results = evaluator.simple_evaluate(
        model="hf",
        model_args=f"pretrained={checkpoint_path}",
        tasks=eval_tasks.split(","),
        limit=2,
        device="cpu"
    )

    save_path = checkpoint_path/"results.json"
    prepare_results(results, save_path, print_results=False)

    print(checkpoint_path/"dump.txt")
    assert save_path.is_file()
    assert results["results"] == {
            'coqa': {
                'alias': 'coqa',
                'em,none': 0.0,
                'em_stderr,none': 0.0,
                'f1,none': 0.0,
                'f1_stderr,none': 0.0
            },
            'hellaswag': {
                'acc,none': 0.0,
                'acc_stderr,none': 0.0,
                'acc_norm,none': 0.5,
                'acc_norm_stderr,none': 0.5,
                'alias': 'hellaswag'
            }
    }
"""
# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
from pathlib import Path
from typing import Optional

from lightning_utilities import is_overridden
from litgpt import Tokenizer
from litgpt.data import LitDataModule
from litgpt.utils import CLI


def prepare(
    data: LitDataModule,
    tokenizer_dir: Optional[Path],
    max_seq_length: Optional[int] = None
) -> None:

    if not is_overridden("prepare_data", data, LitDataModule):
        raise ValueError(
            f"The {type(data).__name__} data module does not support preparing the data in advance."
        )

    tokenizer = Tokenizer(tokenizer_dir)
    data.connect(tokenizer=tokenizer, batch_size=1, max_seq_length=max_seq_length)
    data.prepare_data()


if __name__ == "__main__":
    CLI(prepare)

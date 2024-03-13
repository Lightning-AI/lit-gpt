# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.


from dataclasses import dataclass, field
from pathlib import Path
from litgpt.data.alpaca import Alpaca


_URL = "https://raw.githubusercontent.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/main/data/alpaca_gpt4_data.json"


@dataclass
class AlpacaGPT4(Alpaca):
    """AlpacaGPT4 data module for supervised finetuning."""

    val_split_fraction: float = 0.03847  # to get exactly 2000 test samples,
    """The fraction of the dataset to use for the validation dataset. The rest is used for training."""
    download_dir: Path = Path("./data/alpacagpt4")
    """The directory in which the downloaded datasetgets saved."""
    file_url: str = field(repr=False, default=_URL)
    """The URL from where to download the dataset."""
    file_name: str = field(repr=False, default="alpacagpt4_data_cleaned_archive.json")
    """The name of the dataset file to download."""

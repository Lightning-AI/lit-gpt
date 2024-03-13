# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.


from dataclasses import dataclass, field
from pathlib import Path
from litgpt.data.alpaca import Alpaca
from litgpt.data import SFTDataset


@dataclass
class Alpaca2k(Alpaca):
    """Alpaca2k data module for supervised finetuning."""

    val_split_fraction: float = 0.05  # to get exactly 100 validation samples,
    """The fraction of the dataset to use for the validation dataset. The rest is used for training."""
    download_dir: Path = Path("./data/alpaca2k")
    """The directory in which the downloaded datasetgets saved."""
    repo_id: str = field(repr=False, default="mhenrichsen/alpaca_2k_test")
    """The URL from where to download the dataset."""
    file_name: str = field(repr=False, default="alpaca2k_data_cleaned_archive.json")
    """The name of the dataset file to download."""

    def prepare_data(self) -> None:
        from datasets import load_dataset

        load_dataset(self.repo_id, cache_dir=self.download_dir)

    def setup(self, stage: str = "") -> None:
        from datasets import load_dataset

        dataset = load_dataset(self.repo_id, cache_dir=self.download_dir)

        train_validation_split = dataset["train"].train_test_split(test_size=self.val_split_fraction, seed=self.seed)
        train_data = train_validation_split["train"]
        test_data = train_validation_split["test"]

        self.train_dataset = SFTDataset(
            data=train_data,
            tokenizer=self.tokenizer,
            prompt_style=self.prompt_style,
            max_seq_length=self.max_seq_length,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
        )
        self.test_dataset = SFTDataset(
            data=test_data,
            tokenizer=self.tokenizer,
            prompt_style=self.prompt_style,
            max_seq_length=self.max_seq_length,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
        )


# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, Tuple, Any

import torch
from torch.utils.data import random_split, DataLoader

from litgpt import PromptStyle
from litgpt.data import SFTDataset, get_sft_collate_fn, LitDataModule
from litgpt.tokenizer import Tokenizer


@dataclass
class JSON(LitDataModule):
    """Loads JSON or JSONL data for supervised finetuning."""

    json_path: Path
    """A path to a JSON file or a directory with `train.json` and `val.json` containing the data. 
    The file(s) should contain a list of samples (dicts). Each dict must have the keys 'instruction' and 'output', 
    and can optionally have a key 'input' (see Alpaca)."""
    mask_prompt: bool = False
    """Whether to mask the prompt section from the label (with ``ignore_index``)."""
    val_split_fraction: Optional[float] = None
    """The fraction of the dataset to use for the validation dataset. The rest is used for training.
    Only applies if you passed in a single file to `json_path`."""
    prompt_style: Union[str, PromptStyle] = "alpaca"
    """The style to apply to instruction prompts. See `litgpt.prompts` for a list of available styles."""
    ignore_index: int = -100
    """The index to use for elements to be ignored in the label."""
    seed: int = 42
    """The random seed for creating the train/val splits and shuffling the dataset."""
    num_workers: int = 4
    """How many DataLoader processes to use for loading."""

    tokenizer: Optional[Tokenizer] = field(default=None, init=False, repr=False)
    batch_size: int = field(default=1, init=False, repr=False)
    max_seq_length: int = field(default=-1, init=False, repr=False)
    train_dataset: Optional[SFTDataset] = field(default=None, init=False, repr=False)
    val_dataset: Optional[SFTDataset] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.json_path.is_dir() and self.val_split_fraction is not None:
            raise ValueError(
                "If `json_path` is a directory, it must contain 'train.json' and 'val.json' files and"
                f" hence `val_split_fraction` should not be set. Got `{self.val_split_fraction=}`."
            )
        if not self.json_path.exists():
            raise FileNotFoundError(
                "The `json_path` must be a file or a directory containing 'train.json' and 'val.json' files,"
                f" but '{self.json_path!s}' does not exist."
            )
        if isinstance(self.prompt_style, str):
            self.prompt_style = PromptStyle.from_name(self.prompt_style)

    def connect(
        self,
        tokenizer: Optional[Tokenizer] = None,
        batch_size: int = 1,
        max_seq_length: Optional[int] = None
    ) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_length = -1 if max_seq_length is None else max_seq_length

    def setup(self, stage: str = "") -> None:
        train_data, test_data = self.get_splits()

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

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
            num_workers=self.num_workers,
            collate_fn=get_sft_collate_fn(max_seq_length=self.max_seq_length, ignore_index=self.ignore_index)
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=get_sft_collate_fn(max_seq_length=self.max_seq_length, ignore_index=self.ignore_index)
        )

    def get_splits(self) -> Tuple:
        # A single file (gets split into train and test)
        if self.json_path.is_file():
            data = load_split(self.json_path)

            # Partition the dataset into train and test
            train_data, test_data = random_split(
                data,
                [1.0 - self.val_split_fraction, self.val_split_fraction],
                generator=torch.Generator().manual_seed(self.seed)
            )
            return train_data, test_data

        # A directory containing train.json and val.json
        if (train_file := self.find_split("train")) and (val_file := self.find_split("val")):
            train_data = load_split(train_file)
            test_data = load_split(val_file)
            return train_data, test_data

        raise FileNotFoundError(
            "The `json_path` must be a file or a directory containing 'train.json' and 'val.json' files."
        )

    def find_split(self, split_name: str) -> Optional[Path]:
        for suffix in (".json", ".jsonl"):
            if (file := self.json_path / f"{split_name}{suffix}").is_file():
                return file
        return None


def load_split(json_path: Path) -> Any:
    if json_path.suffix == ".json":
        with open(json_path, "r", encoding="utf-8") as file:
            return json.load(file)
    if json_path.suffix == ".jsonl":
        with open(json_path, "r", encoding="utf-8") as file:
            return [json.loads(line) for line in file]
    else:
        raise ValueError(f"Unsupported file format: {json_path.suffix}. Expected `.json` or `.jsonl`.")

# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import torch
from torch.utils.data import DataLoader

from litgpt import PromptStyle
from litgpt.data import SFTDataset, get_sft_collate_fn, LitDataModule
from litgpt.data.alpaca import download_if_missing
from litgpt.tokenizer import Tokenizer


_URL = "https://raw.githubusercontent.com/akoksal/LongForm/main/dataset"


@dataclass
class LongForm(LitDataModule):
    """LongForm data module for supervised finetuning."""

    mask_prompt: bool = False
    """Whether to mask the prompt section from the label (with ``ignore_index``)."""
    prompt_style: Union[str, PromptStyle] = "longform"
    """The style to apply to instruction prompts. See `litgpt.prompts` for a list of available styles."""
    ignore_index: int = -100
    """The index to use for elements to be ignored in the label."""
    seed: int = 42
    """The random seed for shuffling the dataset."""
    num_workers: int = 4
    """How many DataLoader processes to use for loading."""
    download_dir: Path = Path("./data/longform")
    """The directory in which the downloaded dataset gets saved."""

    tokenizer: Optional[Tokenizer] = field(default=None, init=False, repr=False)
    batch_size: int = field(default=1, init=False, repr=False)
    max_seq_length: int = field(default=-1, init=False, repr=False)
    train_dataset: Optional[SFTDataset] = field(default=None, init=False, repr=False)
    test_dataset: Optional[SFTDataset] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
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

    def prepare_data(self) -> None:
        self.download_dir.mkdir(parents=True, exist_ok=True)
        download_if_missing(self.download_dir / "train.json", f"{_URL}/train.json")
        download_if_missing(self.download_dir / "val.json", f"{_URL}/val.json")

    def train_dataloader(self):
        return self._dataloader("train")

    def val_dataloader(self):
        return self._dataloader("val")

    def _dataloader(self, split: str) -> DataLoader:
        with open(self.download_dir / f"{split}.json", "r", encoding="utf-8") as file:
            data = json.load(file)

        dataset = SFTDataset(
            data=data,
            tokenizer=self.tokenizer,
            prompt_style=self.prompt_style,
            max_seq_length=self.max_seq_length,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
            transform=_transform,
        )
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=(split == "train"),
            generator=torch.Generator().manual_seed(self.seed),
            num_workers=self.num_workers,
            collate_fn=get_sft_collate_fn(max_seq_length=self.max_seq_length, ignore_index=self.ignore_index)
        )


def _transform(item: dict) -> dict:
    item["instruction"] = item.pop("input")
    return item

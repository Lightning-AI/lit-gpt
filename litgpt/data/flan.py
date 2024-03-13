# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Set, Union

import torch
from torch.utils.data import DataLoader

from litgpt import PromptStyle
from litgpt.data import SFTDataset, get_sft_collate_fn, LitDataModule
from litgpt.data.alpaca import download_if_missing
from litgpt.tokenizer import Tokenizer

_URL = "https://huggingface.co/datasets/Muennighoff/flan/resolve/main"


# TODO: Including all subsets, FLAN is too large to be loaded in memory. Switch the implementation to cache
#   on disk or use Lightning Data
@dataclass
class FLAN(LitDataModule):
    """FLAN data module for supervised finetuning."""

    mask_prompt: bool = False
    """Whether to mask the prompt section from the label (with ``ignore_index``)."""
    prompt_style: Union[str, PromptStyle] = "flan"
    """The style to apply to instruction prompts. See `litgpt.prompts` for a list of available styles."""
    ignore_index: int = -100
    """The index to use for elements to be ignored in the label."""
    seed: int = 42
    """The random seed for shuffling the dataset."""
    num_workers: int = 4
    """How many DataLoader processes to use for loading."""
    download_dir: Path = Path("./data/flan")
    """The directory in which the downloaded dataset gets saved."""
    url: str = _URL
    """The URL from where to download the dataset."""
    subsets: Optional[str] = None
    """A comma separated list of subsets to use. If None, all subsets are used."""

    tokenizer: Optional[Tokenizer] = field(default=None, init=False, repr=False)
    batch_size: int = field(default=1, init=False, repr=False)
    max_seq_length: int = field(default=-1, init=False, repr=False)
    train_dataset: Optional[SFTDataset] = field(default=None, init=False, repr=False)
    test_dataset: Optional[SFTDataset] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if isinstance(self.prompt_style, str):
            self.prompt_style = PromptStyle.from_name(self.prompt_style)

        supported_subsets = _supported_subsets()
        if self.subsets is not None:
            self.subsets = self.subsets.split(",")
            for subset in self.subsets:
                if subset not in supported_subsets:
                    raise ValueError(f"{subset} not in {supported_subsets}")
        else:
            self.subsets = list(supported_subsets)

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
        for subset in self.subsets:
            for split in ("train", "test"):
                data_file_path = self.download_dir / f"{subset}_{split}.jsonl"
                data_file_url = f"{self.url}/{split}/{subset}_{split}.jsonl"
                download_if_missing(data_file_path, data_file_url)

    def train_dataloader(self):
        return self._dataloader("train")

    def val_dataloader(self):
        return self._dataloader("test")

    def _dataloader(self, split: str) -> DataLoader:
        data = []
        for subset in self.subsets:
            data_file_path = self.download_dir / f"{subset}_{split}.jsonl"
            data.extend(load_jsonl(data_file_path))

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


def load_jsonl(filename: Path) -> List[Dict[str, str]]:
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def _transform(item: dict) -> dict:
    item["instruction"] = item.pop("inputs")
    item["output"] = item.pop("targets")
    return item


def _supported_subsets() -> Set[str]:
    return {
        "aeslc_10templates",
        "ag_news_subset_10templates",
        "anli_r1_10templates",
        "anli_r2_10templates",
        "anli_r3_10templates",
        "arc_challenge_10templates",
        "arc_easy_10templates",
        "bool_q_10templates",
        "cb_10templates",
        "cnn_dailymail_10templates",
        "cola_10templates",
        "common_gen_10templates",
        "copa_10templates",
        "coqa_10templates",
        "cosmos_qa_10templates",
        "dart_10templates",
        "definite_pronoun_resolution_10templates",
        "drop_10templates",
        "e2e_nlg_10templates",
        "fix_punct_10templates",
        "gigaword_10templates",
        "glue_mrpc_10templates",
        "glue_qqp_10templates",
        "hellaswag_10templates",
        "imdb_reviews_10templates",
        "math_dataset_10templates",
        "mnli_matched_10templates",
        "mnli_mismatched_10templates",
        "multi_news_10templates",
        "multirc_10templates",
        "natural_questions_10templates",
        "openbookqa_10templates",
        "opinion_abstracts_idebate_10templates",
        "opinion_abstracts_rotten_tomatoes_10templates",
        "para_crawl_enes_10templates",
        "paws_wiki_10templates",
        "piqa_10templates",
        "qnli_10templates",
        "quac_10templates",
        "record_10templates",
        "rte_10templates",
        "samsum_10templates",
        "sentiment140_10templates",
        "snli_10templates",
        "squad_v1_10templates",
        "squad_v2_10templates",
        "sst2_10templates",
        "story_cloze_10templates",
        "stsb_10templates",
        "trec_10templates",
        "trivia_qa_10templates",
        "true_case_10templates",
        "web_nlg_en_10templates",
        "wic_10templates",
        "wiki_lingua_english_en_10templates",
        "wmt14_enfr_10templates",
        "wmt16_translate_csen_10templates",
        "wmt16_translate_deen_10templates",
        "wmt16_translate_fien_10templates",
        "wmt16_translate_roen_10templates",
        "wmt16_translate_ruen_10templates",
        "wmt16_translate_tren_10templates",
        "wnli_10templates",
        "word_segment_10templates",
        "wsc_10templates",
        "yelp_polarity_reviews_10templates",
    }

# Copyright (c) OpenMMLab. All rights reserved.

import random

import numpy as np
import torch
from datasets import load_dataset

from xtuner._lite import get_logger

logger = get_logger()


class HuggingfaceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path,
        sample_ratio=1.0,
        tokenize_fn=None,
        cache_dir=None,
        max_length=None,
    ):
        super().__init__()

        dataset = load_dataset(path)["train"]
        self.tokenize_fn = tokenize_fn
        if self.tokenize_fn:
            dataset = dataset.map(self.tokenize_fn, num_proc=8)

        if sample_ratio != 1:
            ori_samples = len(dataset)
            target_samples = int(sample_ratio * ori_samples)
            indices = random.choices([i for i in range(ori_samples)], k=target_samples)
            dataset = dataset.select(indices)

        self.dataset = dataset.filter(
            lambda data: data["num_tokens"] <= max_length, num_proc=4
        )

    @property
    def num_tokens(self):
        return np.array(self.dataset["num_tokens"])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        """Returns a dict containing packed data in the given item.

        Args:
            item: An index to retrieve packed data.

        Returns:
            A dict including packed input_ids, labels, and cumulative_len.
        """
        return self.dataset[item]


if __name__ == "__main__":
    from transformers import AutoTokenizer

    from xtuner._lite.algorithms.sft import SftTokenizeFunction
    from xtuner._lite.chat import CHAT_TEMPLATE_MAP
    from xtuner._lite.datasets import load_datasets

    tokenizer = AutoTokenizer.from_pretrained(
        "internlm/internlm3-8b-instruct", use_fast=False, padding_side="right"
    )

    chat_template = CHAT_TEMPLATE_MAP["internlm2"]

    tokenize_fn = SftTokenizeFunction(
        tokenizer, chat_template, "trl-internal-testing/hh-rlhf-trl-style"
    )

    dataset = HuggingfaceDataset(
        "trl-internal-testing/hh-rlhf-trl-style", tokenize_fn=tokenize_fn
    )
    breakpoint()
    dataset = load_datasets(
        "trl-internal-testing/hh-rlhf-trl-style",
        sources="huggingface",
        map_fns=tokenize_fn,
    )

    breakpoint()

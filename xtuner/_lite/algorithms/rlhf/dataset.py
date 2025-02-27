# Copyright (c) OpenMMLab. All rights reserved.
import json
from dataclasses import asdict
from typing import List

import numpy as np
import torch
from torch import nn

from ..sft import SftCollator
from .trajectory import Trajectory


class InferDataset(torch.utils.data.Dataset):
    def __init__(self, prompts, responses):
        super().__init__()

        assert len(prompts) == len(responses)
        self.prompts = prompts
        self.responses = responses

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, item):
        prompt = self.prompts[item]
        response = self.responses[item]
        num_prefill_tokens = len(prompt)

        input_ids = prompt + response
        labels = [-100] * (num_prefill_tokens - 1) + response + [-100]

        return {"input_ids": input_ids, "labels": labels, "num_tokens": len(input_ids)}


class TrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self, score_ranges=(-5, 5), score_normalize=True):
        super().__init__()

        self.score_ranges = score_ranges

        self.score_normalize = score_normalize

        if self.score_normalize:
            self.bn = nn.BatchNorm1d(1, momentum=None, affine=False)
        else:
            self.bn = None

        self._num_action_tokens = 0
        self._num_total_tokens = 0
        self._trajectories = []

        self._current_mean = 0

    @property
    def running_mean(self):
        return self.bn.running_mean.item()

    @property
    def current_mean(self):
        return self._current_mean

    @property
    def num_action_tokens(self):
        return self._num_action_tokens.item()

    @property
    def num_total_tokens(self):
        return self._num_total_tokens

    def update(self, trajectories: List[Trajectory]) -> None:
        trajectories = [asdict(traj) for traj in trajectories]
        rewards = [data["score"] for data in trajectories]

        for i in range(len(trajectories)):
            trajectories[i]["ori_score"] = trajectories[i]["score"]

        rewards = torch.tensor(rewards)

        self._current_mean = rewards.mean().item()

        rewards = rewards.clip(self.score_ranges[0], self.score_ranges[1])

        if self.score_normalize:
            self.bn.train()
            _ = self.bn(rewards.unsqueeze(-1))
            self.bn.eval()
            rewards = self.bn(rewards.unsqueeze(-1))

        for i in range(len(trajectories)):
            trajectories[i]["score"] = rewards[i].item()

            _prompt_ids = trajectories[i]["prompt_ids"]
            _response_ids = trajectories[i]["response_ids"]

            trajectories[i]["input_ids"] = _prompt_ids + _response_ids
            trajectories[i]["labels"] = (
                [-100] * (len(_prompt_ids) - 1) + _response_ids + [-100]
            )
            trajectories[i]["num_tokens"] = len(trajectories[i]["input_ids"])

        num_total_tokens = 0
        num_action_tokens = 0
        for data in trajectories:
            labels = np.array(data["labels"])
            num_total_tokens += labels.size
            num_action_tokens += (labels >= 0).sum()

        self._num_action_tokens = num_action_tokens
        self._num_total_tokens = num_total_tokens

        self._trajectories = trajectories

    def dump_jsonl(self, path, tokenizer, debug=False):
        with open(path, "w", encoding="utf8") as f:
            for data in self._trajectories:
                json_line = {
                    "num_tokens": data["num_tokens"],
                    "reward": data["ori_reward"],
                    "sequence": tokenizer.decode(data["input_ids"]),
                }

                if debug:
                    json_line["input_ids"] = data["input_ids"]
                    json_line["labels"] = data["labels"]

                json_str = json.dumps(json_line, ensure_ascii=False)
                f.write(json_str + "\n")

    def __len__(self):
        return len(self._trajectories)

    def __getitem__(self, item):
        return self._trajectories[item]


class TrajectoryCollator(SftCollator):
    def __call__(self, instances):
        data = super().__call__(instances)

        data["advantages"] = [item["advantages"] for item in instances]
        data["returns"] = [item["returns"] for item in instances]
        data["old_logprobs"] = [item["old_logprobs"] for item in instances]
        data["ref_logprobs"] = [item["ref_logprobs"] for item in instances]
        data["old_values"] = [item["old_values"] for item in instances]

        return data


class GRPOTrajectoryCollator(SftCollator):
    def __call__(self, instances):
        data = super().__call__(instances)

        data["advantages"] = [item["advantages"] for item in instances]
        data["old_logprobs"] = [item["old_logprobs"] for item in instances]
        data["ref_logprobs"] = [item["ref_logprobs"] for item in instances]

        return data

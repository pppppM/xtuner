# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import dataclass
from typing import List, Optional

import torch

from .loss import compute_advantages_and_returns, compute_kl_rewards


@dataclass
class Trajectory:
    prompt_ids: torch.LongTensor
    response_ids: torch.LongTensor
    old_logprobs: torch.Tensor
    ref_logprobs: torch.Tensor
    score: torch.Tensor
    advantages: torch.Tensor
    kl_rewards: Optional[torch.Tensor] = None
    old_values: Optional[torch.Tensor] = None
    returns: Optional[torch.Tensor] = None


def collect_ppo_trajectories(
    prompt_ids: List[torch.LongTensor],
    response_ids: List[torch.LongTensor],
    old_logprobs: List[torch.Tensor],
    ref_logprobs: List[torch.Tensor],
    old_values: List[torch.Tensor],
    scores: List[torch.Tensor],
    kl_coef: float,
    gamma: float,
    gae_lambda: float,
):
    assert (
        len(prompt_ids)
        == len(response_ids)
        == len(old_logprobs)
        == len(ref_logprobs)
        == len(old_values)
        == len(scores)
    )
    trajectories = []
    for i in range(len(old_logprobs)):
        _kl_rewards = compute_kl_rewards(
            old_logprobs[i], ref_logprobs[i], scores[i], kl_coef
        )
        _advantages, _returns = compute_advantages_and_returns(
            old_values[i], _kl_rewards, gamma, gae_lambda
        )

        assert (
            _returns.shape == old_values[i].shape
        ), f"{_returns.shape} and { old_values[i].shape}"

        trajectory = Trajectory(
            prompt_ids=prompt_ids[i],
            response_ids=response_ids[i],
            old_logprobs=old_logprobs[i],
            ref_logprobs=ref_logprobs[i],
            old_values=old_values[i],
            score=scores[i],
            kl_rewards=_kl_rewards,
            advantages=_advantages,
            returns=_returns,
        )

        trajectories.append(trajectory)

    return trajectories


def collect_trajectories(
    prompt_ids: List[torch.LongTensor],
    response_ids: List[torch.LongTensor],
    old_logprobs: List[torch.Tensor],
    ref_logprobs: List[torch.Tensor],
    scores: List[torch.Tensor],
    kl_coef: float,
    prompt_repeat_k: int,
    alg: str,
):
    assert (
        len(prompt_ids)
        == len(response_ids)
        == len(old_logprobs)
        == len(ref_logprobs)
        == len(scores)
    )
    assert alg in ["rloo", "grpo"]
    _rewards = torch.tensor(scores).reshape(-1, prompt_repeat_k).T
    if alg == "rloo":
        baseline = (_rewards.sum(0) - _rewards) / (prompt_repeat_k - 1)
        advantages = _rewards - baseline
        advantages = advantages.T.flatten()
    elif alg == "grpo":
        advantages = (_rewards - _rewards.mean(0)) / (_rewards.std(0) + 1e-8)
        advantages = advantages.T.flatten()

    trajectories = []
    for i in range(len(old_logprobs)):
        _kl_rewards = compute_kl_rewards(
            old_logprobs[i], ref_logprobs[i], scores[i], kl_coef
        )
        
        trajectory = Trajectory(
            prompt_ids=prompt_ids[i],
            response_ids=response_ids[i],
            old_logprobs=old_logprobs[i],
            ref_logprobs=ref_logprobs[i],
            score=scores[i],
            kl_rewards=_kl_rewards,
            advantages=advantages[i],
        )

        trajectories.append(trajectory)

    return trajectories

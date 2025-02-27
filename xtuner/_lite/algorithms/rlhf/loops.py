# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from typing import Optional

import torch
from torch import distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from xtuner._lite.accelerate import profile_time_and_memory, unpack_sequence
from xtuner._lite.algorithms.sft import SftCollator
from xtuner._lite.parallel import ParallelSampler
from xtuner._lite.patches.base import PatchedLLM

from .dataset import TrajectoryCollator, TrajectoryDataset
from .loss import CriticLoss, PPOPolicyLoss


@profile_time_and_memory("[Actor Update]")
def update_actor_loop(
    actor: PatchedLLM,
    loss_fn: PPOPolicyLoss,
    optimizer: Optimizer,
    max_grad_norm: int,
    dataset: TrajectoryDataset,
    micro_batch_size: int,
    global_batch_size: int,
    dp_mesh: DeviceMesh,
    sp_mesh: Optional[DeviceMesh] = None,
    optimizer_step_times: int = 1,
    device: str = "cuda",
    debug: bool = False,
):
    log_dict = OrderedDict()

    # Count the total number of tokens used for training PPO on all ranks
    # It is necessary for `per-token` loss, otherwise the number of tokens
    # for each backward is unbalanced.
    global_action_tokens = dataset.num_action_tokens
    loss_factor = dp_mesh.size() / global_action_tokens

    sum_rank_loss = 0

    sampler = ParallelSampler(dataset, dp_mesh, global_batch_size)

    dataloader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        num_workers=0,
        collate_fn=TrajectoryCollator(pack_batch=True),
        shuffle=False,
        sampler=sampler,
        persistent_workers=False,
    )

    assert len(dataloader) % optimizer_step_times == 0

    dataiter = iter(dataloader)

    actor.train()
    for step in range(optimizer_step_times):
        if actor.fsdp_config.cpu_offload:
            actor.patched_model.set_is_last_backward(False)
            actor.patched_model.set_requires_gradient_sync(False)

        for _iter in range(len(dataloader) // optimizer_step_times):
            trajectory = next(dataiter)

            input_ids = trajectory["input_ids"].to(device)
            num_tokens = trajectory["num_tokens"].to(device)
            assert input_ids.numel() == num_tokens.sum()

            # labels are shifted
            labels = trajectory["labels"].to(device)

            old_logprobs = trajectory["old_logprobs"]
            advantages = trajectory["advantages"]

            num_tokens = num_tokens.clone().tolist()

            cu_seq_lens = (
                torch.cumsum(torch.IntTensor([0] + num_tokens), dim=0).to(device).int()
            )
            _position_ids = [torch.arange(num) for num in num_tokens]
            position_ids = torch.cat(_position_ids, dim=0).to(device).unsqueeze_(0)

            packed_logits = actor(
                input_ids=input_ids,
                position_ids=position_ids,
                use_cache=False,
                cu_seq_lens_q=cu_seq_lens,
                cu_seq_lens_k=cu_seq_lens,
                max_length_q=max(num_tokens),
                max_length_k=max(num_tokens),
                sequence_parallel_mesh=sp_mesh,
            ).logits

            packed_logprobs = actor.gather_logprobs(
                packed_logits, labels.clip(0), sp_mesh
            )

            logprobs = unpack_sequence(packed_logprobs, num_tokens, dim=1)
            unpacked_labels = unpack_sequence(labels, num_tokens, dim=1)

            _policy_losses = []
            for i in range(micro_batch_size):
                assert unpacked_labels[i].numel() == num_tokens[i]
                # from the last prefill token, to the second-to-last token (excluding the eos token)
                _num_action_tokens = (unpacked_labels[i] >= 0).sum()

                _logprobs = logprobs[i][0, -_num_action_tokens - 1 : -1]
                _old_logprobs = old_logprobs[i].to(device)
                _advantages = advantages[i].to(device)

                # When using per token loss, it is necessary to calibrate the
                # loss based on the global number of action tokens.
                _policy_loss = loss_fn(
                    _logprobs, _old_logprobs, _advantages, loss_factor
                )
                _policy_losses.append(_policy_loss)

            policy_loss = sum(_policy_losses)

            if actor.fsdp_config.cpu_offload and iter == len(dataloader) - 1:
                actor.patched_model.set_is_last_backward(True)
                actor.patched_model.set_requires_gradient_sync(True)

            policy_loss.backward()
            sum_rank_loss += policy_loss.detach()

        grad_norm = actor.clip_grad_norm(max_grad_norm)
        log_dict["actor_grad_norm"] = grad_norm

        optimizer.step()
        optimizer.zero_grad()

    reduced_loss = dist.nn.all_reduce(sum_rank_loss, group=dp_mesh.get_group())
    reduced_loss.div_(dp_mesh.size())
    log_dict["actor_loss(reduced)"] = reduced_loss.item()

    if debug:
        log_dict["actor_loss(rank)"] = sum_rank_loss.item()

    return log_dict


@profile_time_and_memory("[Critic Update]")
def update_critic_loop(
    critic: PatchedLLM,
    optimizer: Optimizer,
    max_grad_norm: int,
    loss_fn: CriticLoss,
    dataset: TrajectoryDataset,
    global_batch_size: int,
    micro_batch_size: int,
    dp_mesh: DeviceMesh,
    sp_mesh: Optional[DeviceMesh] = None,
    optimizer_step_times: Optional[int] = -1,
    device: str = "cuda",
    debug: bool = False,
):
    log_dict = OrderedDict()

    sum_rank_loss = 0

    # Count the total number of tokens used for training PPO on all ranks
    # It is necessary for `per-token` loss, otherwise the number of tokens
    # for each backward is unbalanced.
    global_action_tokens = dataset.num_action_tokens
    loss_factor = dp_mesh.size() / global_action_tokens

    sampler = ParallelSampler(dataset, dp_mesh, global_batch_size)

    dataloader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        num_workers=0,
        collate_fn=TrajectoryCollator(pack_batch=True),
        shuffle=False,
        sampler=sampler,
        persistent_workers=False,
    )

    assert len(dataloader) % optimizer_step_times == 0

    dataiter = iter(dataloader)

    critic.train()
    for step in range(optimizer_step_times):
        if critic.fsdp_config.cpu_offload and critic.fsdp_config.reshard_after_forward:
            critic.patched_model.set_is_last_backward(False)
            critic.patched_model.set_requires_gradient_sync(False)

        for _iter in range(len(dataloader) // optimizer_step_times):
            trajectory = next(dataiter)
            input_ids = trajectory["input_ids"].to(device)
            num_tokens = trajectory["num_tokens"].to(device)
            assert input_ids.numel() == num_tokens.sum()

            # labels are shifted
            ppo_labels = trajectory["labels"].to(device)

            old_values = trajectory["old_values"]
            returns = trajectory["returns"]

            num_tokens = num_tokens.tolist()

            cu_seq_lens = (
                torch.cumsum(torch.IntTensor([0] + num_tokens), dim=0).to(device).int()
            )
            _position_ids = [torch.arange(num) for num in num_tokens]
            position_ids = torch.cat(_position_ids, dim=0).to(device).unsqueeze_(0)

            packed_values = critic(
                input_ids=input_ids,
                position_ids=position_ids,
                use_cache=False,
                cu_seq_lens_q=cu_seq_lens,
                cu_seq_lens_k=cu_seq_lens,
                max_length_q=max(num_tokens),
                max_length_k=max(num_tokens),
                sequence_parallel_mesh=sp_mesh,
            ).logits

            packed_values = critic.gather_logprobs(
                packed_values, torch.zeros_like(ppo_labels), sp_mesh
            )
            critic_values = unpack_sequence(packed_values, num_tokens, dim=1)

            unpacked_labels = unpack_sequence(ppo_labels, num_tokens, dim=1)

            _critic_losses = []
            for i in range(micro_batch_size):
                # from the last prefill token, to the second-to-last token (excluding the eos token)
                _num_action_tokens = (unpacked_labels[i] >= 0).sum()

                _values = critic_values[i][0, -_num_action_tokens - 1 : -1]
                _old_values = old_values[i].to(device)
                _returns = returns[i].to(device)

                _critic_loss = loss_fn(_values, _old_values, _returns, loss_factor)
                _critic_losses.append(_critic_loss)

            critic_loss = sum(_critic_losses)

            if (
                critic.fsdp_config.cpu_offload
                and critic.fsdp_config.reshard_after_forward
                and iter == len(dataloader) - 1
            ):
                critic.patched_model.set_is_last_backward(True)
                critic.patched_model.set_requires_gradient_sync(True)

            critic_loss.backward()

            sum_rank_loss += critic_loss.detach()

        grad_norm = critic.clip_grad_norm(max_grad_norm)
        log_dict["critic_grad_norm"] = grad_norm.item()

        optimizer.step()
        optimizer.zero_grad()

    reduced_loss = dist.nn.all_reduce(sum_rank_loss, group=dp_mesh.get_group())
    reduced_loss.div_(dp_mesh.size())
    log_dict["critic_loss(reduced)"] = reduced_loss.item()

    if debug:
        log_dict["critic_loss(rank)"] = sum_rank_loss

    return log_dict


@profile_time_and_memory("[Compute Logprobs]")
@torch.no_grad
def compute_logprobs_loop(
    model: PatchedLLM,
    dataset: TrajectoryDataset,
    micro_batch_size: int,
    sp_mesh: Optional[DeviceMesh] = None,
    device: str = "cuda",
):
    dataloader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        num_workers=0,
        collate_fn=SftCollator(pack_batch=True),
        shuffle=False,
        persistent_workers=False,
    )

    model.eval()

    results = []
    for sequences in dataloader:
        # labels are shifted
        labels = sequences["labels"].to(device)
        input_ids = sequences["input_ids"].to(device)
        num_tokens = sequences["num_tokens"].to(device)

        num_tokens = num_tokens.tolist()

        cu_seq_lens = (
            torch.cumsum(torch.IntTensor([0] + num_tokens), dim=0).to(device).int()
        )
        _position_ids = [torch.arange(num) for num in num_tokens]
        position_ids = torch.cat(_position_ids, dim=0).to(device).unsqueeze_(0)

        with torch.no_grad():
            packed_logits = model(
                input_ids=input_ids,
                position_ids=position_ids,
                use_cache=False,
                cu_seq_lens_q=cu_seq_lens,
                cu_seq_lens_k=cu_seq_lens,
                max_length_q=max(num_tokens),
                max_length_k=max(num_tokens),
                sequence_parallel_mesh=sp_mesh,
            ).logits

        packed_logprobs = model.gather_logprobs(packed_logits, labels.clip(0), sp_mesh)

        unpacked_labels = unpack_sequence(labels, num_tokens, dim=1)
        logprobs = unpack_sequence(packed_logprobs, num_tokens, dim=1)

        for i in range(micro_batch_size):
            assert unpacked_labels[i].numel() == num_tokens[i]

            _num_action_tokens = (unpacked_labels[i] >= 0).sum()
            _logprobs = logprobs[i][0, -_num_action_tokens - 1 : -1]
            results.append(_logprobs)

    return results


@profile_time_and_memory("[Compute Scores]")
@torch.no_grad
def compute_scores_loop(
    model: PatchedLLM,
    dataset: TrajectoryDataset,
    micro_batch_size: int,
    sp_mesh: Optional[DeviceMesh] = None,
    device: str = "cuda",
):
    dataloader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        num_workers=0,
        collate_fn=SftCollator(pack_batch=True),
        shuffle=False,
        persistent_workers=False,
    )

    model.eval()

    results = []
    for sequences in dataloader:
        # labels are shifted
        labels = sequences["labels"].to(device)
        input_ids = sequences["input_ids"].to(device)
        num_tokens = sequences["num_tokens"].to(device)

        num_tokens = num_tokens.tolist()

        cu_seq_lens = (
            torch.cumsum(torch.IntTensor([0] + num_tokens), dim=0).to(device).int()
        )
        _position_ids = [torch.arange(num) for num in num_tokens]
        position_ids = torch.cat(_position_ids, dim=0).to(device).unsqueeze_(0)

        with torch.no_grad():
            packed_logits = model(
                input_ids=input_ids,
                position_ids=position_ids,
                use_cache=False,
                cu_seq_lens_q=cu_seq_lens,
                cu_seq_lens_k=cu_seq_lens,
                max_length_q=max(num_tokens),
                max_length_k=max(num_tokens),
                sequence_parallel_mesh=sp_mesh,
            ).logits

        packed_scores = model.gather_logprobs(
            packed_logits, torch.zeros_like(labels), sp_mesh
        )

        unpacked_labels = unpack_sequence(labels, num_tokens, dim=1)
        scores = unpack_sequence(packed_scores, num_tokens, dim=1)

        for i in range(micro_batch_size):
            assert unpacked_labels[i].numel() == num_tokens[i]
            _score = scores[i][0, -1]
            results.append(_score)

    return results


@profile_time_and_memory("[Compute Values]")
@torch.no_grad
def compute_values_loop(
    model: PatchedLLM,
    dataset: TrajectoryDataset,
    micro_batch_size: int,
    sp_mesh: Optional[DeviceMesh] = None,
    device: str = "cuda",
):
    dataloader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        num_workers=0,
        collate_fn=SftCollator(pack_batch=True),
        shuffle=False,
        persistent_workers=False,
    )

    model.eval()

    results = []
    for sequences in dataloader:
        # labels are shifted
        labels = sequences["labels"].to(device)
        input_ids = sequences["input_ids"].to(device)
        num_tokens = sequences["num_tokens"].to(device)

        num_tokens = num_tokens.tolist()

        cu_seq_lens = (
            torch.cumsum(torch.IntTensor([0] + num_tokens), dim=0).to(device).int()
        )
        _position_ids = [torch.arange(num) for num in num_tokens]
        position_ids = torch.cat(_position_ids, dim=0).to(device).unsqueeze_(0)

        with torch.no_grad():
            packed_logits = model(
                input_ids=input_ids,
                position_ids=position_ids,
                use_cache=False,
                cu_seq_lens_q=cu_seq_lens,
                cu_seq_lens_k=cu_seq_lens,
                max_length_q=max(num_tokens),
                max_length_k=max(num_tokens),
                sequence_parallel_mesh=sp_mesh,
            ).logits

        packed_values = model.gather_logprobs(
            packed_logits, torch.zeros_like(labels), sp_mesh
        )

        unpacked_labels = unpack_sequence(labels, num_tokens, dim=1)
        values = unpack_sequence(packed_values, num_tokens, dim=1)

        for i in range(micro_batch_size):
            assert unpacked_labels[i].numel() == num_tokens[i]
            _num_action_tokens = (unpacked_labels[i] >= 0).sum()
            _values = values[i][0, -_num_action_tokens - 1 : -1]
            results.append(_values)

    return results

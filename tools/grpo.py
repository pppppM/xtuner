# Copyright (c) OpenMMLab. All rights reserved.
import os
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Optional

import torch
import torch.distributed as dist
import yaml
from cyclopts import App
from mmengine.runner import set_random_seed
from mmengine.utils import mkdir_or_exist
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader
from transformers import AutoTokenizer

from xtuner._lite import get_device, get_logger, get_torch_device_module, log_format
from xtuner._lite.accelerate import profile_time_and_memory, unpack_sequence
from xtuner._lite.algorithms.rlhf import (
    InferDataset,
    PPOPolicyLoss,
    TrajectoryDataset,
    collect_trajectories,
    compute_logprobs_loop,
    compute_scores_loop,
    update_actor_loop,
)
from xtuner._lite.algorithms.sft import SftCollator, SftTokenizeFunction
from xtuner._lite.arguments import (
    DatasetArguments,
    EngineArguments,
    ModelArguments,
    PPOArguments,
    RolloutArguments,
)
from xtuner._lite.chat import CHAT_TEMPLATE_MAP
from xtuner._lite.datasets import load_datasets
from xtuner._lite.parallel import ParallelSampler, setup_parallel
from xtuner._lite.patches import AutoPatchForCausalLM, AutoPatchForReward

app = App()
logger = get_logger()
DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()

torch._dynamo.config.cache_size_limit = 16384


def is_interval(step, total_steps, interval):
    return (step + 1) % interval == 0 or (step + 1) == total_steps


# @logger.catch
@app.default
def grpo(
    config: str,
    /,
    *,
    actor: Optional[ModelArguments] = None,
    reference: Optional[ModelArguments] = None,
    reward: Optional[ModelArguments] = None,
    rollout: Optional[RolloutArguments] = None,
    data: Optional[DatasetArguments] = None,
    pt_data: Optional[DatasetArguments] = None,
    engine: Optional[EngineArguments] = None,
):
    with open(config) as file:
        config = yaml.safe_load(file)

    config: PPOArguments = PPOArguments.model_validate(config)

    setup_parallel()
    set_random_seed(config.engine.seed)

    rank = dist.get_rank()

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    objects = [timestamp]
    dist.broadcast_object_list(objects, src=0)
    timestamp = objects[0]

    config.engine.work_dir = os.path.join(config.engine.work_dir, timestamp)
    mkdir_or_exist(config.engine.work_dir)

    if config.engine.debug or rank == 0:
        log_file = os.path.join(config.engine.work_dir, f"rank{rank}.log")
        logger.add(log_file, format=log_format(rank), backtrace=True, catch=True)

    actor_model = AutoPatchForCausalLM.from_arguments(config.actor)
    ref_model = AutoPatchForCausalLM.from_arguments(config.reference)
    reward_model = AutoPatchForReward.from_arguments(config.reward)

    actor_sp_mesh = actor_model.sequence_parallel_mesh
    actor_dp_mesh = actor_model.data_parallel_mesh

    actor_dp_size = actor_dp_mesh.size()

    tokenizer = AutoTokenizer.from_pretrained(
        config.dataset.tokenizer, padding_side="right"
    )

    if config.dataset.chat_template is not None:
        chat_template = CHAT_TEMPLATE_MAP[config.dataset.chat_template]
    else:
        chat_template = actor_model.chat_template

    stop_token_ids = chat_template.get_stop_token_ids(tokenizer)

    with profile_time_and_memory("[Dataset & Dataloader]"):
        tokenize_fns = []
        for _format in config.dataset.formats:
            # If your data format is not in `SUPPORT_DATA_FORMATS`, you should
            # redefine a `tokenize_fn`, defining how to convert a piece of raw
            # data into tokenized data.
            # The tokenized data must include `input_ids`, `labels``,
            # and `num_tokens`.
            tokenize_fn = SftTokenizeFunction(tokenizer, chat_template, _format)
            tokenize_fns.append(tokenize_fn)

        _datasets = load_datasets(
            paths=config.dataset.paths,
            # cache_dir=args.dset_cache_dir,
            file_types=config.dataset.suffixes,
            sources=config.dataset.sources,
            sample_ratios=config.dataset.weights,
            map_fns=tokenize_fns,
            # file_pattern=args.file_pattern,
            max_length=config.dataset.max_length,
        )

        if rank == 0:
            num_samples = sum([len(dset) for dset in _datasets])
            logger.info(f"[Dataset] {num_samples} samples.")

        rollout_dataset = ConcatDataset(_datasets)

        rollout_collator = SftCollator(pack_batch=True)
        rollout_sampler = ParallelSampler(
            rollout_dataset,
            dp_mesh=actor_dp_mesh,
            global_batch_size=config.algorithm.rollout.dataloader.global_batch_size,
            shuffle=True,
        )

        rollout_dataloader = DataLoader(
            rollout_dataset,
            batch_size=config.algorithm.rollout.dataloader.global_batch_size
            // actor_dp_mesh.size(),
            num_workers=config.algorithm.rollout.dataloader.num_workers,
            # Ensure to round up or drop last based on the `ppo_global_batch`,
            # if you want to replace a custom sampler.
            sampler=rollout_sampler,
            collate_fn=rollout_collator,
            persistent_workers=config.algorithm.rollout.dataloader.num_workers > 0,
        )

        if rank == 0 and config.engine.debug:
            logger.info(f"[Dataloader] {len(rollout_dataloader)} batches.")
            _first_batch = [
                rollout_dataset[i]
                for i in range(config.algorithm.rollout.dataloader.micro_batch_size)
            ]
            logger.debug(f"[Dataloader] Training Batch:\n{_first_batch}")

        rollout_iterator = iter(rollout_dataloader)

        trajectory_dataset = TrajectoryDataset(
            score_ranges=(config.algorithm.min_score, config.algorithm.max_score),
            score_normalize=config.algorithm.normalize_score,
        )

    policy_loss_fn = PPOPolicyLoss(loss_type="per_token")

    actor_params = [p for p in actor_model.parameters() if p.requires_grad]
    actor_optimizer = AdamW(
        actor_params,
        lr=config.algorithm.update_actor.optim.lr,
        weight_decay=config.algorithm.update_actor.optim.weight_decay,
    )

    if config.algorithm.total_steps > 0:
        total_steps = config.algorithm.total_steps
    else:
        total_steps = len(rollout_dataloader)

    if config.engine.checkpoint_interval == -1:
        checkpoint_interval = total_steps
    elif config.engine.checkpoint_interval < 1:
        checkpoint_interval = int(total_steps * config.engine.checkpoint_interval)
    else:
        checkpoint_interval = int(config.engine.checkpoint_interval)

    start_step = 0
    start_train_t = time.time()
    DEVICE_MODULE.empty_cache()
    DEVICE_MODULE.reset_peak_memory_stats()
    max_memory = DEVICE_MODULE.max_memory_allocated()

    logger.info(
        "[Train] Begin Train Loop. The current GPU memory is "
        f"{(max_memory / 1024**3):.1f}GB"
    )

    for step in range(start_step, total_steps):
        step_start_t = time.time()

        # Stage 1,  Actor Model Generation
        with profile_time_and_memory("[Rollout]"):
            data = next(rollout_iterator)
            prompt_ids = unpack_sequence(
                data["input_ids"].to(DEVICE), data["num_tokens"]
            )
            # repeat prompt for k times, AAAABBBBCCCC
            prompt_ids = [p for p in prompt_ids for _ in range(config.algorithm.rollout.prompt_repeat_k)]

            # # During the generation stage, sequence parallelism was not used,
            # # even when the sp size is greater than 1.
            # # Per sp rank processes different prompts in parallel.
            response_ids = actor_model.generate(
                prompt_ids,
                stop_token_ids,
                max_length=config.algorithm.rollout.max_length,
                max_batch_size=len(prompt_ids),
                max_prefill_batch=len(prompt_ids),
                max_new_tokens=config.algorithm.rollout.max_new_tokens,
                do_sample=config.algorithm.rollout.do_sample,
                top_k=config.algorithm.rollout.top_k,
                top_p=config.algorithm.rollout.top_p,
                temperature=config.algorithm.rollout.temperature,
                # cuda_graph=args.cuda_graph,
            )

        prompt_ids = [pmt_ids[0].tolist() for pmt_ids in prompt_ids]

        # Stage 2,  Infer
        infer_dataset = InferDataset(prompt_ids, response_ids)

        old_logprobs = compute_logprobs_loop(
            actor_model,
            infer_dataset,
            config.algorithm.update_actor.dataloader.micro_batch_size,
            actor_model.sequence_parallel_mesh,
        )

        ref_logprobs = compute_logprobs_loop(
            ref_model,
            infer_dataset,
            config.algorithm.update_actor.dataloader.micro_batch_size,
            ref_model.sequence_parallel_mesh,
        )

        scores = compute_scores_loop(
            reward_model,
            infer_dataset,
            config.algorithm.update_actor.dataloader.micro_batch_size,
            reward_model.sequence_parallel_mesh,
        )

        assert len(old_logprobs) == len(ref_logprobs) == len(scores)

        trajectories = collect_trajectories(
            prompt_ids,
            response_ids,
            old_logprobs,
            ref_logprobs,
            scores,
            config.algorithm.kl_coef,
            config.algorithm.rollout.prompt_repeat_k,
            config.algorithm.alg
        )

        # Stage 3, PPO
        _global_trajectories = [None] * actor_dp_size

        dist.all_gather_object(
            _global_trajectories, trajectories, group=actor_dp_mesh.get_group()
        )

        global_trajectories = []
        for _rank_trajectories in _global_trajectories:
            global_trajectories.extend(_rank_trajectories)

        trajectory_dataset.update(global_trajectories)

        for _ in range(config.algorithm.update_actor.epochs):
            update_actor_log = update_actor_loop(
                actor=actor_model,
                loss_fn=policy_loss_fn,
                optimizer=actor_optimizer,
                max_grad_norm=config.algorithm.update_actor.optim.max_grad_norm,
                dataset=trajectory_dataset,
                micro_batch_size=config.algorithm.update_actor.dataloader.micro_batch_size,
                global_batch_size=config.algorithm.update_actor.dataloader.global_batch_size,
                optimizer_step_times=1,
                dp_mesh=actor_dp_mesh,
                sp_mesh=actor_sp_mesh,
            )

        log_dict = OrderedDict()
        log_dict["step"] = step + 1
        log_dict["actor_lr"] = (
            config.algorithm.update_actor.optim.lr
        )

        log_dict.update(update_actor_log)

        max_memory = DEVICE_MODULE.max_memory_allocated()
        log_dict["max_memory(GB)"] = max_memory / 1024**3

        step_time = time.time() - step_start_t
        eta = step_time * (total_steps - step)
        eta = timedelta(seconds=int(eta))
        log_dict["step_time(s)"] = step_time
        log_dict["eta"] = eta.seconds

        # if rank == 0:
        #     with open(
        #         os.path.join(config.engine.work_dir, f"rank{rank}.log.jsonl"), "a"
        #     ) as f:
        #         f.write(json.dumps(log_dict, ensure_ascii=False) + "\n")

        if is_interval(step, total_steps, config.engine.log_interval):
            log_str = f"[Step {step+1}]"
            for key, value in log_dict.items():
                log_str += f"  {key}: {value}"
            logger.info(log_str)

        if is_interval(step, total_steps, checkpoint_interval):
            DEVICE_MODULE.empty_cache()

            num_digits = len(str(abs(total_steps)))
            work_dir = config.engine.work_dir
            # ckpt_dir = os.path.join(work_dir, f"ckpt-{step+1:0{num_digits}}")
            hf_dir = os.path.join(work_dir, f"hf-{step+1:0{num_digits}}")

            with profile_time_and_memory("[Checkpoint]"):
                actor_model.save_pretrained(hf_dir)
                if rank == 0:
                    tokenizer.save_pretrained(hf_dir)

    train_cost_time = time.time() - start_train_t
    logger.success(f"[Train] Cost {timedelta(seconds=int(train_cost_time))}")
    # ------------------------    Training  End  ---------------------------- #


if __name__ == "__main__":
    app()

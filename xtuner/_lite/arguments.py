# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Literal, Optional

from cyclopts import Parameter
from pydantic import BaseModel
from typing_extensions import Annotated


class OptimArguments(BaseModel):
    lr: Annotated[float, Parameter(help="Learning rate for optimization")] = 1e-5
    max_grad_norm: Annotated[
        float, Parameter(help="Maximum gradient norm for gradient clipping")
    ] = 1.0
    warmup_ratio: Annotated[
        float, Parameter(help="Ratio of warmup steps to total training steps")
    ] = 0.0
    weight_decay: Annotated[
        float, Parameter(help="Weight decay coefficient for L2 regularization")
    ] = 0.01


class DatasetArguments(BaseModel):
    tokenizer: Annotated[
        Optional[str], Parameter(help="Path or name of the tokenizer to use")
    ] = None
    chat_template: Annotated[
        Optional[str], Parameter(help="Template format for chat conversations")
    ] = None
    paths: Annotated[
        List[str], Parameter(help="List of dataset paths or names to use for training")
    ] = ""
    weights: Annotated[
        List[float], Parameter(help="Sampling weights for each dataset")
    ] = [1.0]
    suffixes: Annotated[
        List[Literal[".jsonl", ".json"]],
        Parameter(help="Allowed file extensions for dataset files"),
    ] = [".jsonl"]
    sources: Annotated[
        List[Literal["local", "huggingface"]],
        Parameter(help="Source of datasets: local files or HuggingFace hub"),
    ] = ["huggingface"]
    formats: Annotated[
        List[Literal["openai", "alpaca", "trl-internal-testing/hh-rlhf-trl-style"]],
        Parameter(help="Format of the dataset: OpenAI or other styles"),
    ] = ["openai"]
    max_length: Annotated[
        int, Parameter(help="Maximum sequence length for input tokens")
    ] = 8192


class DataloaderArguments(BaseModel):
    num_workers: Annotated[
        int, Parameter(help="Total batch size across all devices")
    ] = 1
    global_batch_size: Annotated[
        int, Parameter(help="Total batch size across all devices")
    ] = 16
    micro_batch_size: Annotated[
        int, Parameter(help="Batch size per device, -1 for auto-calculation")
    ] = -1
    max_length: Annotated[
        int, Parameter(help="Maximum sequence length for input tokens")
    ] = 8192
    packing_type: Annotated[
        Optional[Literal["soft", "hard"]],
        Parameter(help="Type of sequence packing strategy"),
    ] = "soft"
    packing_level: Annotated[
        Optional[Literal["global", "file"]],
        Parameter(help="Level at which to apply sequence packing"),
    ] = "global"


class ModelArguments(BaseModel):
    model: Annotated[
        str, Parameter(help="Model identifier or path")
    ] = "internlm/internlm3-8b-instruct"
    dtype: Annotated[
        Literal["fp16", "bf16", "auto"],
        Parameter(help="Data type for model weights and computation"),
    ] = "auto"
    cpu_offload: Annotated[
        bool, Parameter(help="Enable CPU offloading for memory optimization")
    ] = False
    reshard_after_forward: Annotated[
        bool, Parameter(help="Reshard model parameters after forward pass")
    ] = True
    compile: Annotated[
        bool, Parameter(help="Enable model compilation for faster inference")
    ] = False
    recompute: Annotated[
        float, Parameter(help="Gradient checkpointing ratio for memory optimization")
    ] = 1.0
    requires_grad: Annotated[
        bool, Parameter(help="Enable gradient computation for model parameters")
    ] = True
    sp_size: Annotated[int, Parameter(help="Sequence parallelism size")] = 1
    tp_size: Annotated[int, Parameter(help="Tensor parallelism size")] = 1
    ep_size: Annotated[int, Parameter(help="Expert parallelism size")] = 1
    mesh_prefix: Annotated[
        str,
        Parameter(help="Prefix for device mesh configuration in distributed training"),
    ] = "default"


class EngineArguments(BaseModel):
    work_dir: Annotated[
        str, Parameter(help="Directory for saving checkpoints and logs")
    ] = "./work_dir"
    resume: Annotated[
        bool, Parameter(help="Resume training from last checkpoint")
    ] = False
    seed: Annotated[int, Parameter(help="Random seed for reproducibility")] = 0
    log_interval: Annotated[int, Parameter(help="Interval between logging updates")] = 1
    checkpoint_interval: Annotated[
        int, Parameter(help="Interval between saving checkpoints, -1 to disable")
    ] = -1
    checkpoint_drop_optimizer: Annotated[
        bool, Parameter(help="Exclude optimizer state from checkpoints")
    ] = True
    debug: Annotated[
        bool, Parameter(help="Enable debug mode with additional logging")
    ] = False


class RolloutArguments(BaseModel):
    dataloader: Annotated[
        DataloaderArguments,
        Parameter(help="Configuration for data loading during rollout"),
    ] = DataloaderArguments()
    max_length: Annotated[
        int, Parameter(help="Maximum total length of input and generated sequences")
    ] = 2048
    max_new_tokens: Annotated[
        int, Parameter(help="Maximum number of new tokens to generate")
    ] = 1024
    max_concurrency: Annotated[
        int, Parameter(help="Maximum number of concurrent generation tasks")
    ] = 128
    max_pefill_tokens: Annotated[
        int, Parameter(help="Maximum number of tokens for prefill phase")
    ] = 8192
    do_sample: Annotated[
        bool, Parameter(help="Enable sampling-based generation")
    ] = True
    top_k: Annotated[
        int, Parameter(help="Top-k filtering parameter, -1 to disable")
    ] = -1
    top_p: Annotated[float, Parameter(help="Top-p nucleus sampling parameter")] = 0.9
    n: Annotated[int, Parameter(help="Number of generation samples per input")] = 1
    temperature: Annotated[
        float, Parameter(help="Sampling temperature for generation")
    ] = 1.0


class UpdateActorArguments(BaseModel):
    optim: Annotated[
        Optional[OptimArguments], Parameter(help="Optimization configuration")
    ] = None
    dataloader: Annotated[
        DataloaderArguments,
        Parameter(help="Configuration for data loading during actor updates"),
    ]
    epochs: Annotated[int, Parameter(help="Number of training epochs for actor")] = 1
    mini_batch_size: Annotated[
        int, Parameter(help="Mini-batch size for actor updates, -1 for auto")
    ] = -1
    warmup_steps: Annotated[
        int, Parameter(help="Number of warmup steps for actor training")
    ] = 0


class UpdateCriticArguments(BaseModel):
    optim: Annotated[
        Optional[OptimArguments], Parameter(help="Optimization configuration")
    ] = None
    dataloader: Annotated[
        DataloaderArguments,
        Parameter(help="Configuration for data loading during critic updates"),
    ]
    epochs: Annotated[int, Parameter(help="Number of training epochs for critic")] = 1
    mini_batch_size: Annotated[
        int, Parameter(help="Mini-batch size for critic updates, -1 for auto")
    ] = -1
    reinitialize: Annotated[
        Optional[float], Parameter(help="Reinitialization ratio for critic head")
    ] = None


class PretrainArguments(BaseModel):
    dataset: Annotated[
        DatasetArguments, Parameter(help="Dataset configuration for pretraining")
    ]
    dataloader: Annotated[
        DataloaderArguments, Parameter(help="Dataloader configuration for pretraining")
    ]
    loss_weight: Annotated[
        float, Parameter(help="Weight factor for pretraining loss")
    ] = 0.05


class PPOAlgorithmArguements(BaseModel):
    kl_coef: Annotated[
        float, Parameter(help="KL divergence coefficient for PPO")
    ] = 0.01
    gamma: Annotated[float, Parameter(help="Discount factor for future rewards")] = 1.0
    gae_lambda: Annotated[
        float, Parameter(help="Lambda parameter for GAE calculation")
    ] = 0.99
    min_score: Annotated[float, Parameter(help="Minimum score for reward scaling")] = -5
    max_score: Annotated[float, Parameter(help="Maximum score for reward scaling")] = 5
    normalize_score: Annotated[
        bool, Parameter(help="Enable score normalization")
    ] = True
    rollout: Annotated[
        RolloutArguments, Parameter(help="Configuration for rollout phase")
    ] = RolloutArguments()
    update_actor: Annotated[
        UpdateActorArguments, Parameter(help="Configuration for actor model updates")
    ]
    update_critic: Annotated[
        UpdateCriticArguments, Parameter(help="Configuration for critic model updates")
    ]
    pretrain: Annotated[
        Optional[PretrainArguments],
        Parameter(help="Configuration for pretraining phase"),
    ] = None
    total_steps: Annotated[
        int,
        Parameter(
            help="Total number of training steps, -1 for automatic determination"
        ),
    ] = -1


class PPOArguments(BaseModel):
    actor: ModelArguments
    reward: ModelArguments
    reference: ModelArguments
    critic: ModelArguments
    algorithm: PPOAlgorithmArguements
    engine: EngineArguments = EngineArguments()
    dataset: DatasetArguments
    pt_dataset: Optional[DatasetArguments] = None

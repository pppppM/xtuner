# Copyright (c) OpenMMLab. All rights reserved.
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.llama import LlamaForCausalLM
from transformers.models.qwen2 import Qwen2ForCausalLM

from xtuner._lite import get_torch_device_module
from xtuner._lite.arguments import ModelArguments
from xtuner._lite.modelings.internlm3 import InternLM3ForCausalLM

from .base import FSDPConfig, PatchedLLM
from .internlm3 import CUDAPatchedInternLM3, MLUPatchedInternLM3
from .llama import CUDAPatchedLlama, MLUPatchedLlama
from .qwen2 import CUDAPatchedQwen2

CUDA_PATCH_MAP = {
    LlamaForCausalLM: CUDAPatchedLlama,
    InternLM3ForCausalLM: CUDAPatchedInternLM3,
    Qwen2ForCausalLM: CUDAPatchedQwen2,
}

MLU_PATCH_MAP = {
    LlamaForCausalLM: MLUPatchedLlama,
    InternLM3ForCausalLM: MLUPatchedInternLM3,
}

REWARD_CONVERT_MAP = {
    "LlamaForSequenceClassification": "LlamaForCausalLM",
    "LlamaForCausalLM": "LlamaForCausalLM",
    "Qwen2ForSequenceClassification": "Qwen2ForCausalLM",
    "Qwen2ForTokenClassification": "Qwen2ForCausalLM",
    "Qwen2ForCausalLM": "Qwen2ForCausalLM",
}


class AutoPatch:
    @classmethod
    def from_causal_lm(
        cls, model, fsdp_config: FSDPConfig, is_reward=False, device_type="cuda"
    ) -> PatchedLLM:
        if device_type == "cuda":
            patch_cls = CUDA_PATCH_MAP[type(model)]
        elif device_type == "mlu":
            patch_cls = MLU_PATCH_MAP[type(model)]
        else:
            raise NotImplementedError

        patched_model = patch_cls(model, fsdp_config, is_reward)

        return patched_model


class AutoPatchForCausalLM:
    @classmethod
    def from_arguments(cls, args: ModelArguments):
        if args.dtype == "auto":
            args.dtype = (
                "bf16" if get_torch_device_module().is_bf16_supported() else "fp16"
            )

        if args.dtype == "fp16":
            dtype = torch.float16
        elif args.dtype == "bf16":
            if get_torch_device_module().is_bf16_supported():
                dtype = torch.bfloat16
            else:
                raise RuntimeError(
                    "The device does not support `bf16`, "
                    "please set `dtype` to `fp16`."
                )
        else:
            raise RuntimeError(
                "`dtype` only supports `fp16`, `bf16` or `auto`, "
                f"but found {args.dtype}."
            )

        with torch.device("meta"):
            model = AutoModelForCausalLM.from_pretrained(
                args.model, attn_implementation="flash_attention_2", torch_dtype=dtype
            )

            if args.requires_grad:
                for module in model.modules():
                    for p_name, param in module.named_parameters(recurse=False):
                        if param.requires_grad:
                            param_fp32 = torch.nn.Parameter(
                                param.to(dtype=torch.float32)
                            )
                            setattr(module, p_name, param_fp32)
            else:
                for param in model.parameters():
                    param.requires_grad = False

        fsdp_config = FSDPConfig(
            tp_size=args.tp_size,
            ep_size=args.ep_size,
            sp_size=args.sp_size,
            reshard_after_forward=args.reshard_after_forward,
            cpu_offload=args.cpu_offload,
            param_dtype=dtype,
            reduce_dtype=dtype,
            torch_compile=args.compile,
            mesh_prefix=args.mesh_prefix,
        )

        return AutoPatch.from_causal_lm(model, fsdp_config, is_reward=False)


class AutoPatchForReward:
    @classmethod
    def from_arguments(cls, args: ModelArguments):
        if args.dtype == "auto":
            args.dtype = (
                "bf16" if get_torch_device_module().is_bf16_supported() else "fp16"
            )

        if args.dtype == "fp16":
            dtype = torch.float16
        elif args.dtype == "bf16":
            if get_torch_device_module().is_bf16_supported():
                dtype = torch.bfloat16
            else:
                raise RuntimeError(
                    "The device does not support `bf16`, "
                    "please set `dtype` to `fp16`."
                )
        else:
            raise RuntimeError(
                "`dtype` only supports `fp16`, `bf16` or `auto`, "
                f"but found {args.dtype}."
            )

        with torch.device("meta"):
            hf_config = AutoConfig.from_pretrained(args.model)

            ori_arch = hf_config.architectures[0]
            if ori_arch not in REWARD_CONVERT_MAP:
                raise NotImplementedError(
                    f"The reward model of type `{ori_arch}` is not supported."
                )

            hf_config.architectures[0] = REWARD_CONVERT_MAP[ori_arch]

            model = AutoModelForCausalLM.from_config(
                hf_config, attn_implementation="flash_attention_2", torch_dtype=dtype
            )

            if ori_arch == "Qwen2ForTokenClassification":
                model.lm_head = torch.nn.Linear(
                    hf_config.hidden_size, hf_config.num_labels
                ).to(dtype)
            else:
                model.lm_head = torch.nn.Linear(
                    hf_config.hidden_size, hf_config.num_labels, bias=False
                ).to(dtype)

            if args.requires_grad:
                for module in model.modules():
                    for p_name, param in module.named_parameters(recurse=False):
                        if param.requires_grad:
                            param_fp32 = torch.nn.Parameter(
                                param.to(dtype=torch.float32)
                            )
                            setattr(module, p_name, param_fp32)
            else:
                for param in model.parameters():
                    param.requires_grad = False

        fsdp_config = FSDPConfig(
            tp_size=args.tp_size,
            ep_size=args.ep_size,
            sp_size=args.sp_size,
            reshard_after_forward=args.reshard_after_forward,
            cpu_offload=args.cpu_offload,
            param_dtype=dtype,
            reduce_dtype=dtype,
            torch_compile=args.compile,
            mesh_prefix=args.mesh_prefix,
        )

        return AutoPatch.from_causal_lm(model, fsdp_config, is_reward=True)

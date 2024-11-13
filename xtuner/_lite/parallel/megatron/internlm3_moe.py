from functools import partial

import torch
from torch import nn
import torch.distributed as dist
from torch.distributed._tensor import Replicate, distribute_tensor, Shard
from torch.distributed.tensor.parallel import (ColwiseParallel,
                                               PrepareModuleInput,
                                               PrepareModuleOutput,
                                               RowwiseParallel,
                                               parallelize_module)

from xtuner._lite import get_logger
from .utils import map_rank0_modules
from ..fsdp.lazy import lazy_init_megatron

logger = get_logger()


def shard_experts_weight(mesh, weight, shard_dim=0):
    world_size = mesh.size()
    chunks = torch.chunk(weight, world_size, dim=shard_dim)
    return chunks[mesh.get_local_rank()]


# @torch.no_grad
# def lazy_init_megatron(module, rank0_map, experts_fsdp_mesh, ep_mesh):
#     device = torch.cuda.current_device()
#     rank = dist.get_rank()

#     if rank == 0 :
#         rank0_module = rank0_map[module]
#         rank0_params = {
#             name: param
#             for name, param in rank0_module.named_parameters(recurse=False)
#         }
#         rank0_buffers = {
#             name: buffer
#             for name, buffer in rank0_module.named_buffers(recurse=False)
#         }
#     else:
#         rank0_params = None
#         rank0_buffers = None 

#     param_shapes = {
#         name : param.shape
#         for name, param in module.named_parameters(recurse=False)
#     }

#     module.to_empty(device=torch.cuda.current_device(), recurse=False)

#     for name, param in module.named_parameters(recurse=False):
#         dtype = param.dtype
#         if rank == 0:
#             rank0_param = rank0_params[name].to(device).to(dtype)
#         else:
#             full_shape = param_shapes[name]
#             rank0_param = torch.zeros(full_shape, dtype=dtype, device=device)

#         mesh = param.device_mesh
#         placements = param.placements

#         dist.broadcast(rank0_param, src=0)

#         if placements == (Shard(0), Shard(0)):
#             param_this_rank = shard_experts_weight(ep_mesh, rank0_param, shard_dim=0)
#             param_this_rank = shard_experts_weight(experts_fsdp_mesh, param_this_rank, shard_dim=0)
#         else:
#             param_this_rank = distribute_tensor(rank0_param, mesh, placements).data.to_local()
#         # param.data.copy_(rank0_dtensor)
#         param.data.to_local().copy_(param_this_rank)
    
#     # FSDP does not shard buffers
#     for name, buffer in module.named_buffers(recurse=False):
#         # dtype = buffer.dtype
#         if rank == 0:
#             rank0_buffer = rank0_buffers[name].to(device).to(buffer.dtype)
#         else:
#             rank0_buffer = torch.empty_like(buffer).to(device)

#         dist.broadcast(rank0_buffer, src=0)
#         buffer.data.copy_(rank0_buffer)


def megatron_internlm3_moe(model,
                       rank0_model,
                       experts_fsdp_mesh,
                       ep_mesh,
                       mp_policy=None,
                       recompute_ratio=1.0,
                       reshard_after_forward=True):
    if experts_fsdp_mesh.get_rank() == 0:
        rank0_map = map_rank0_modules(model, rank0_model)
    else:
        rank0_map = None

    param_init_fn = partial(
        lazy_init_megatron,
        rank0_map=rank0_map,
        dp_mesh=experts_fsdp_mesh,
    )

    from torch.distributed._composable import checkpoint
    from torch.distributed._composable.fsdp import fully_shard
    num_layers = len(model.layers)
    num_recompute_layers = int(num_layers * recompute_ratio)

    for i, block in enumerate(model.layers):

        block.apply(param_init_fn)

        fully_shard(
            block,
            mesh=experts_fsdp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward,
        )

        if i < num_recompute_layers:
            checkpoint(block)
    
    for layer_cur, layer_next in zip(model.layers[:-1], model.layers[1:]):
        layer_cur.set_modules_to_forward_prefetch([layer_next])

    model.embed_tokens.apply(param_init_fn)
    model.norm.apply(param_init_fn)


def megatron_internlm3_moe_casual(model,
                       rank0_model,
                       experts_fsdp_mesh,
                       ep_mesh,
                       mp_policy=None,
                       recompute_ratio=1.0,
                       reshard_after_forward=True):
    megatron_internlm3_moe(
        model.model,
        rank0_model.model if experts_fsdp_mesh.get_rank() == 0 else None,
        experts_fsdp_mesh,
        ep_mesh,
        mp_policy=mp_policy,
        recompute_ratio=recompute_ratio,
        reshard_after_forward=reshard_after_forward
    )

    if experts_fsdp_mesh.get_rank() == 0:
        rank0_map = map_rank0_modules(model, rank0_model)
    else:
        rank0_map = None
    
    param_init_fn = partial(
        lazy_init_megatron,
        rank0_map=rank0_map,
        dp_mesh=experts_fsdp_mesh,
    )

    model.lm_head.apply(param_init_fn)

    from torch.distributed._composable.fsdp import fully_shard
    fully_shard(
        model,
        mesh=experts_fsdp_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=False)
    
    model.set_modules_to_forward_prefetch([model.model.layers[0]])

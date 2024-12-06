from functools import partial
from torch.distributed._composable.fsdp import fully_shard
from xtuner._lite import get_logger
from ..fsdp.lazy import lazy_init_megatron
from .utils import map_rank0_modules
from ..fsdp import checkpoint

logger = get_logger()


def megatron_janus_casual(meta_model,
                          rank0_model,
                          dp_mesh,
                          tp_mesh=None,
                          pp_mesh=None,
                          mp_policy=None,
                          recompute_ratio=1.0,
                          reshard_after_forward=True,
                          freeze_style='mode1'):
    if tp_mesh.size() > 1:
        raise NotImplementedError

    if dp_mesh.get_rank() == 0:
        rank0_map = map_rank0_modules(meta_model, rank0_model)
    else:
        rank0_map = None

    param_init_fn = partial(
        lazy_init_megatron,
        rank0_map=rank0_map,
        dp_mesh=dp_mesh,
        tp_mesh=tp_mesh,
    )

    if freeze_style == 'mode1':
        meta_model.language_model.model.apply(param_init_fn)
        fully_shard(
            meta_model.language_model.model,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward,
        )
        meta_model.language_model.lm_head.apply(param_init_fn)
        fully_shard(
            meta_model.language_model.lm_head,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward,
        )
        meta_model.gen_head.apply(param_init_fn)
        fully_shard(
            meta_model.gen_head,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward,
        )

        model = fully_shard(
            meta_model,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward)  # False is zero2, True is zero3

        model.set_reshard_after_backward(False)

    else:
        raise NotImplementedError
    return model

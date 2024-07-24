import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

_SP_MESH = None
_DP_MESH = None
_SP_GROUP = None
_DP_GROUP = None
_SP_WORLD_SIZE = None
_DP_WORLD_SIZE = None


def init_sp_device_mesh(sp_size):
    world_size = dist.get_world_size()
    assert world_size % sp_size == 0
    dp_size = world_size // sp_size
    device_mesh = init_device_mesh(
        'cuda', (dp_size, sp_size), mesh_dim_names=('dp', 'sp'))

    global _SP_MESH, _DP_MESH
    _SP_MESH = device_mesh['sp']
    _DP_MESH = device_mesh['dp']

    global _SP_GROUP, _DP_GROUP
    _SP_GROUP = device_mesh.get_group('sp')
    _DP_GROUP = device_mesh.get_group('dp')


def get_dp_mesh():
    return _DP_MESH


def get_sp_mesh():
    return _SP_MESH


def get_sp_group():
    return _SP_GROUP


def get_sp_world_size():
    global _SP_WORLD_SIZE
    if _SP_WORLD_SIZE is not None:
        return _SP_WORLD_SIZE
    if not dist.is_initialized() or (_SP_GROUP is None):
        _SP_WORLD_SIZE = 1
    else:
        _SP_WORLD_SIZE = dist.get_world_size(_SP_GROUP)
    return _SP_WORLD_SIZE


def get_dp_world_size():
    global _DP_WORLD_SIZE
    if _DP_WORLD_SIZE is not None:
        return _DP_WORLD_SIZE
    if not dist.is_initialized() or (_DP_GROUP is None):
        _DP_WORLD_SIZE = 1
    else:
        _DP_WORLD_SIZE = dist.get_world_size(_DP_GROUP)
    return _DP_WORLD_SIZE

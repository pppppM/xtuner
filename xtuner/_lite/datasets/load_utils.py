import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import (_get_pg_default_device,
                                                _object_to_tensor,
                                                _tensor_to_object)


def all_to_all_list(object_list, group=None):
    current_device = _get_pg_default_device(group)
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    tensor_list, size_list = zip(
        *
        [_object_to_tensor(obj, current_device, group) for obj in object_list])
    tensor_list = list(tensor_list)
    size_list = torch.cat(size_list)
    buffer = [None] * world_size

    dist.all_gather_object(buffer, size_list, group=group)
    size_this_rank = []
    for size_list in buffer:
        size_this_rank.append(size_list[rank])

    target_tensor_list = [
        torch.empty(size.item(), dtype=torch.uint8, device=current_device)
        for size in size_this_rank
    ]
    dist.all_to_all(target_tensor_list, tensor_list, group=group)

    for i in range(len(target_tensor_list)):
        obj_view = target_tensor_list[i].type(torch.uint8)
        target_tensor_list[i] = _tensor_to_object(obj_view, size_this_rank[i],
                                                  group)

    return target_tensor_list

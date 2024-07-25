import os
import random
import string

import torch
import torch.distributed as dist
from datasets import concatenate_datasets, load_from_disk
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


def assign_files(files):
    if not dist.is_available():
        return list(range(len(files)))

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if rank == 0:

        # Assigned files to each rank based on the file size
        file_sizes = []
        for file in files:
            if os.path.islink(file):
                real_path = os.path.realpath(file)
                file_sizes.append(os.path.getsize(real_path))
            else:
                file_sizes.append(os.path.getsize(file))

        size_order = sorted(
            enumerate(file_sizes), key=lambda x: x[1], reverse=True)
        sorted_indices = [ind_and_size[0] for ind_and_size in size_order]

        per_rank_files = [[] for _ in range(world_size)]
        per_rank_sizes = [0 for _ in range(world_size)]

        for ind in sorted_indices:

            min_size = min(per_rank_sizes)
            target = per_rank_sizes.index(min_size)

            per_rank_files[target].append(ind)
            per_rank_sizes[target] += file_sizes[ind]

        objects = [per_rank_files]
    else:
        objects = [None]
    dist.broadcast_object_list(objects, src=0)
    per_rank_files = objects[0]
    files_this_rank = per_rank_files[rank]
    return files_this_rank


def mktmpdir():

    def mkdir():
        random_string = ''.join(
            random.choices(string.ascii_lowercase + string.digits, k=8))
        path = f'tmp{random_string}'
        while os.path.exists(path):
            random_string = ''.join(
                random.choices(string.ascii_lowercase + string.digits, k=8))
            path = f'tmp{random_string}'
        os.mkdir(path)
        return path

    if not dist.is_available():
        return mkdir()

    if dist.get_rank() == 0:
        path = mkdir()
        objects = [path]
    else:
        objects = [None]
    dist.broadcast_object_list(objects, src=0)

    return objects[0]


def save_part_data(ds, file_idx, num_files, root_dir):
    digits = len(str(abs(num_files)))
    file_id = (f'file-{file_idx+1:0{digits}}-of-'
               f'{num_files:0{digits}}')

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    digits = len(str(world_size))
    rank_id = (f'file-{rank+1:0{digits}}-of-'
               f'{world_size:0{digits}}')
    path = os.path.join(root_dir, file_id, rank_id)
    ds.save_to_disk(path)


def load_whole_data(file_idx, num_files, root_dir):
    world_size = dist.get_world_size()

    digits = len(str(abs(num_files)))
    file_id = (f'file-{file_idx+1:0{digits}}-of-'
               f'{num_files:0{digits}}')

    path = os.path.join(root_dir, file_id)
    items = os.listdir(path)
    folders = [item for item in items if os.path.isdir(item)]
    assert len(folders) == world_size, f'{folders}'

    ds_list = []
    for folder in folders:
        path = os.path.join(root_dir, file_id, folder)
        ds_list.append(load_from_disk(path))
    ds = concatenate_datasets(ds_list)
    return ds


def barrier():
    if not dist.is_available():
        return

    rank = dist.get_rank()
    if rank == 0:
        objects = [1]
    else:
        objects = [None]

    dist.broadcast_object_list(objects, src=0)
    return

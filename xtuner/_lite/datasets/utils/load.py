# Copyright (c) OpenMMLab. All rights reserved.
import os
import re

from xtuner._lite import get_logger

from ..hf_hub import HuggingfaceDataset
from ..json import JsonDataset
from ..jsonl import JsonlDataset

logger = get_logger()

DATASET_CLS_MAP = {".jsonl": JsonlDataset, ".json": JsonDataset}


def load_hf_dataset(
    path, split="train", sample_ratio=1.0, cache_dir=None, map_fn=None, max_length=None
):
    return HuggingfaceDataset(path, sample_ratio, map_fn, cache_dir, max_length)


def load_local_datasets(
    paths,
    file_types,
    file_pattern=None,
    cache_dir=None,
    sample_ratios=1.0,
    map_fns=None,
    max_length=None,
):
    if isinstance(paths, str):
        paths = [paths]

    if isinstance(sample_ratios, (tuple, list)):
        if len(sample_ratios) == 1:
            sample_ratios = list(sample_ratios) * len(paths)

        if len(sample_ratios) != len(paths):
            raise RuntimeError(
                f"There are {len(paths)} paths, but only "
                f"{len(sample_ratios)} sample ratios were set."
            )

    if map_fns is None:
        map_fns = [None] * len(paths)

    if isinstance(map_fns, (tuple, list)):
        if len(map_fns) == 1:
            map_fns = list(map_fns) * len(paths)

        if len(map_fns) != len(paths):
            raise RuntimeError(
                f"There are {len(paths)} paths, but only"
                f"{len(map_fns)} map fns were set."
            )

    files = []
    file_sample_ratios = []
    file_map_fns = []

    for pid, path in enumerate(paths):
        if os.path.isdir(path):
            dir_files = []
            for root, dirs, _files in os.walk(path, followlinks=True):
                dirs.sort()
                for relative_path in sorted(_files):
                    suffix = os.path.splitext(relative_path)[-1]
                    absolute_path = os.path.join(root, relative_path)
                    if file_pattern is not None:
                        if bool(re.match(file_pattern, absolute_path)):
                            dir_files.append(absolute_path)
                    elif suffix in file_types:
                        dir_files.append(absolute_path)

            _num_dir_files = len(dir_files)
            if _num_dir_files == 0:
                raise RuntimeError(
                    f"There are no files with the suffix {file_types}" f"in `{path}`."
                )

            logger.info(f"Found {len(dir_files)} files in {path}")
            files.extend(dir_files)
            file_sample_ratios.extend([sample_ratios[pid]] * _num_dir_files)
            file_map_fns.extend([map_fns[pid]] * _num_dir_files)

        elif os.path.isfile(path):
            files.append(path)
            file_sample_ratios.append(sample_ratios[pid])
            file_map_fns.append(map_fns[pid])

        else:
            raise RuntimeError(f"`{path}` not found.")

    num_files = len(files)

    datasets = []
    for i in range(num_files):
        _path = files[i]
        _ratio = file_sample_ratios[i]
        _map_fn = file_map_fns[i]
        _suffix = os.path.splitext(_path)[-1]

        dataset_cls = DATASET_CLS_MAP[_suffix]
        _dataset = dataset_cls(_path, _ratio, _map_fn, cache_dir, max_length)
        datasets.append(_dataset)

    return datasets


def load_datasets(
    paths,
    sources="local",
    sample_ratios=1.0,
    file_types=DATASET_CLS_MAP.keys(),
    file_pattern=None,
    cache_dir=None,
    map_fns=None,
    max_length=None,
):
    if isinstance(paths, str):
        paths = [paths]

    num_paths = len(paths)

    if isinstance(sample_ratios, (float, int)):
        sample_ratios = [sample_ratios] * num_paths

    if isinstance(sample_ratios, (tuple, list)):
        if len(sample_ratios) == 1:
            sample_ratios = list(sample_ratios) * num_paths

        if len(sample_ratios) != num_paths:
            raise RuntimeError(
                f"There are {num_paths} paths, but only "
                f"{len(sample_ratios)} sample ratios were set."
            )

    if isinstance(sources, str):
        sources = [sources]

    if isinstance(sources, (tuple, list)):
        if len(sources) == 1:
            sources = list(sources) * num_paths

        if len(sources) != num_paths:
            raise RuntimeError(
                f"There are {num_paths} paths, but only "
                f"{len(sources)} sources were set."
            )

    if not isinstance(map_fns, (tuple, list)):
        map_fns = [map_fns] * num_paths

    if isinstance(map_fns, (tuple, list)):
        if len(map_fns) == 1:
            map_fns = list(map_fns) * num_paths

        if len(map_fns) != num_paths:
            raise RuntimeError(
                f"There are {num_paths} paths, but only"
                f"{len(map_fns)} map fns were set."
            )

    local_inds = [i for i, src in enumerate(sources) if src == "local"]
    local_paths = [paths[ind] for ind in local_inds]
    local_map_fns = [map_fns[ind] for ind in local_inds]
    local_sample_ratios = [sample_ratios[ind] for ind in local_inds]

    hf_inds = [i for i, src in enumerate(sources) if src == "huggingface"]
    hf_paths = [paths[ind] for ind in hf_inds]
    hf_map_fns = [map_fns[ind] for ind in hf_inds]
    hf_sample_ratios = [sample_ratios[ind] for ind in hf_inds]

    datasets = []
    if len(local_inds):
        local_datasets = load_local_datasets(
            local_paths,
            file_types,
            file_pattern,
            cache_dir,
            local_sample_ratios,
            local_map_fns,
            max_length,
        )
        datasets.extend(local_datasets)

    if len(hf_inds):
        for i in range(len(hf_inds)):
            dset = load_hf_dataset(
                hf_paths[i],
                sample_ratio=hf_sample_ratios[i],
                map_fn=hf_map_fns[i],
                cache_dir=cache_dir,
                max_length=max_length,
            )
            datasets.append(dset)

    return datasets


def load_ms_dataset():
    pass

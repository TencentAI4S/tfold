# -*- coding: utf-8 -*-
# Copyright (c) 2022, Tencent Inc. All rights reserved.
from functools import partial
from typing import List, Union, Tuple

import numpy as np
import torch


def to(tensor,
       device=None,
       dtype=None,
       non_blocking=False):
    if isinstance(tensor, (list, tuple)):
        new_tensors = []
        for t in tensor:
            new_tensors.append(to(t,
                                  device=device,
                                  dtype=dtype,
                                  non_blocking=non_blocking))
        return new_tensors

    elif isinstance(tensor, dict):
        new_dict = {}
        for name, value in tensor.items():
            new_dict[name] = to(value,
                                device=device,
                                dtype=dtype,
                                non_blocking=non_blocking)
        return new_dict
    elif isinstance(tensor, torch.Tensor):
        return tensor.to(device=device,
                         dtype=dtype,
                         non_blocking=non_blocking)
    elif isinstance(tensor, np.ndarray):
        return torch.tensor(tensor, dtype=dtype, device=device)
    else:
        return tensor


def to_tensor(data, dtype=None, device=None):
    return to(data, dtype=dtype, device=device)


def to_device(tensor, device, non_blocking=False):
    return to(tensor, device, non_blocking=non_blocking)


def clone(data):
    """Clone the data object w/ gradients detached.

    Args:
        data: source data object (list / dict / torch.Tensor / etc.)

    Returns:
        data: cloned data object (list / dict / torch.Tensor / etc.)
    """

    if isinstance(data, list):
        return [clone(x) for x in data]
    if isinstance(data, dict):
        return {k: clone(v) for k, v in data.items()}
    if isinstance(data, torch.Tensor):
        return data.detach().clone()

    return data


def permute_final_dims(tensor: torch.Tensor,
                       inds: Union[List, Tuple]):
    zero_index = -1 * len(inds)  # -2
    first_inds = list(range(len(tensor.shape[:zero_index])))
    orders = first_inds + [zero_index + i for i in inds]
    return tensor.permute(orders)


def flatten_final_dims(t: torch.Tensor, num_dims: int):
    return t.reshape(t.shape[:-num_dims] + (-1,))


def dict_map(fn, dic, leaf_type):
    new_dict = {}
    for k, v in dic.items():
        if type(v) is dict:
            new_dict[k] = dict_map(fn, v, leaf_type)
        else:
            new_dict[k] = tree_map(fn, v, leaf_type)

    return new_dict


tensor_dict_map = partial(dict_map, leaf_type=torch.Tensor)


def tree_map(fn, tree, leaf_type):
    if isinstance(tree, dict):
        return dict_map(fn, tree, leaf_type)
    elif isinstance(tree, list):
        return [tree_map(fn, x, leaf_type) for x in tree]
    elif isinstance(tree, tuple):
        return tuple([tree_map(fn, x, leaf_type) for x in tree])
    elif isinstance(tree, leaf_type):
        return fn(tree)
    else:
        print(type(tree))
        raise ValueError('Not supported')


tensor_tree_map = partial(tree_map, leaf_type=torch.Tensor)


def cdist(x1, x2=None):
    """Calculate the pairwise distance matrix.

    Args:
        x1: input tensor of size N x D or B x N x D
        x2: (optional) input tensor of size M x D or B x M x D

    Returns:
        dist_tns: pairwise distance of size N x M or B x N x M

    Note:
    * If <x2> is not provided, then pairwise distance will be computed within <x1>.
    * The matrix multiplication approach will not be used to avoid the numerical stability issue.
    """
    x2 = x1 if x2 is None else x2

    # recursively call if the batch dimension is missing
    if (x1.ndim == 2) and (x2.ndim == 2):
        return cdist(x1.unsqueeze(dim=0), x2.unsqueeze(dim=0))[0]

    # validate inputs
    assert (x1.ndim == 3) and (x2.ndim == 3)
    assert (x1.shape[0] == x2.shape[0]) and (x1.shape[2] == x2.shape[2])

    # calculate the pairwise distance matrix
    dist = torch.cdist(x1, x2, compute_mode='donot_use_mm_for_euclid_dist')

    return dist

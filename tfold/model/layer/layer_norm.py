# -*- coding: utf-8 -*-
# Copyright (c) 2023, Tencent Inc. All rights reserved.
import logging
import numbers
from typing import Union, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_shape_t = Union[int, List[int], torch.Size]

try:
    from apex.normalization.fused_layer_norm import fused_layer_norm, fused_layer_norm_affine
    logging.warning('using apex fused layer norm')
except:
    fused_layer_norm = None
    fused_layer_norm_affine = None


class LayerNorm(nn.Module):
    """LayerNorm with optional bias. See description of nn.LayerNorm"""
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(self,
                 normalized_shape: _shape_t,
                 eps: float = 1e-5,
                 elementwise_affine: bool = True,
                 bias: bool = True,
                 device=None,
                 dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]

        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
            self.bias = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs)) if bias else None
        else:
            assert not bias, f'when elementwise_affine = False, bias item is None'
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if fused_layer_norm is None or torch.jit.is_tracing() or torch.jit.is_scripting() or not input.is_cuda:
            return F.layer_norm(
                input, self.normalized_shape, self.weight, self.bias, self.eps)

        if self.elementwise_affine:
            return fused_layer_norm_affine(input, self.weight, self.bias, self.normalized_shape, self.eps)
        else:
            return fused_layer_norm(input, self.normalized_shape, self.eps)

    def extra_repr(self) -> str:
        return '{normalized_shape}, eps={eps}, ' \
               'elementwise_affine={elementwise_affine}'.format(**self.__dict__)

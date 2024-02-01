# -*- coding: utf-8 -*-
# Copyright (c) 2023, Tencent Inc. All rights reserved.
from functools import partialmethod
from typing import Optional, List

import torch
from torch import nn

from tfold.model.layer import Linear, LayerNorm, Attention
from tfold.model.utils.chunk_utils import chunk_layer
from tfold.utils.tensor import permute_final_dims


class TriangleAttention(nn.Module):
    """
    Args:
        head_dim: Overall hidden channel dimension (not per-head)
        num_heads: Number of attention heads
    """
    def __init__(
            self,
            dim,
            c_hidden,
            num_heads,
            starting=True,
            inf=1e9
    ):
        super().__init__()
        self.dim = dim
        self.c_hidden = c_hidden
        self.num_heads = num_heads
        self.starting = starting
        self.inf = inf
        self.layer_norm = LayerNorm(self.dim)
        self.linear = Linear(self.dim, self.num_heads, bias=False, init='normal')
        self.mha = Attention(self.dim, self.dim, self.dim, self.c_hidden, self.num_heads)

    @torch.jit.ignore
    def _chunk(self,
               x: torch.Tensor,
               biases: List[torch.Tensor],
               chunk_size: int,
               inplace_safe: bool = False,
               ) -> torch.Tensor:
        "triangle! triangle!"
        mha_inputs = {
            'q_x': x,
            'kv_x': x,
            'biases': biases,
        }
        return chunk_layer(
            self.mha,
            mha_inputs,
            chunk_size=chunk_size,
            no_batch_dims=len(x.shape[:-2]),
            _out=x if inplace_safe else None,
        )

    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                chunk_size: Optional[int] = None,
                inplace_safe: bool = False,
                ) -> torch.Tensor:
        """
        Args:
            x: [*, I, J, C_in] input tensor (e.g. the pair representation, bs x seq_len x seq_len x c_z)
            mask: None or [*, I, J]
        Returns:
            [*, I, J, C_in] output tensor
        """
        if mask is None:
            mask = x.new_ones(
                x.shape[:-1],
            )

        if not self.starting:
            x = x.transpose(-2, -3)
            mask = mask.transpose(-1, -2)

        x = self.layer_norm(x)
        # [*, I, 1, 1, J]
        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]

        # [*, H, I, J]
        triangle_bias = permute_final_dims(self.linear(x), [2, 0, 1])
        # [*, 1, H, I, J]
        triangle_bias = triangle_bias.unsqueeze(-4)

        biases = [mask_bias, triangle_bias]

        if chunk_size is not None:
            x = self._chunk(
                x,
                biases,
                chunk_size,
                inplace_safe=inplace_safe,
            )
        else:
            x = self.mha(
                q_x=x,
                kv_x=x,
                biases=biases
            )

        if not self.starting:
            x = x.transpose(-2, -3)

        return x


# Implements Algorithm 13
TriangleAttentionStartingNode = TriangleAttention


class TriangleAttentionEndingNode(TriangleAttention):
    """
    Implements Algorithm 14.
    """
    __init__ = partialmethod(TriangleAttention.__init__, starting=False)

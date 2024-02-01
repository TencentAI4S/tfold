# -*- coding: utf-8 -*-
# Copyright (c) 2023, Tencent Inc. All rights reserved.
import logging
from typing import Optional, Tuple, List

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from tfold.model.layer import Linear
from tfold.utils.tensor import permute_final_dims, flatten_final_dims

try:
    from deepspeed.ops.deepspeed4science import DS4Sci_EvoformerAttention

    logging.warning(f'using DS4Sci_EvoformerAttention')
except ImportError:
    DS4Sci_EvoformerAttention = None


@torch.jit.ignore
def _deepspeed_evo_attn(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        biases: List[torch.Tensor],
):
    """""
    Compute attention using the DeepSpeed DS4Sci_EvoformerAttention kernel.

    Args:
        q:
            [*, Q, H, C_hidden] query data
        k:
            [*, K, H, C_hidden] key data
        v:
            [*, V, H, C_hidden] value data
        biases:
            List of biases that broadcast to [*, H, Q, K]
    """

    def reshape_dims(x):
        no_batch_dims = len(x.shape[:-3])
        if no_batch_dims < 2:
            return x.reshape(*((1,) * (2 - no_batch_dims) + x.shape))
        if no_batch_dims > 2:
            return x.reshape(*((x.shape[0], -1) + x.shape[-3:]))
        return x

    # Reshape tensors to match expected input shape [B, N, Q/K, H, C_hidden]
    # for DS4Sci_EvoformerAttention() by adding or flattening batch dims as needed.
    orig_shape = q.shape
    if len(orig_shape[:-3]) != 2:
        q = reshape_dims(q)
        k = reshape_dims(k)
        v = reshape_dims(v)
        biases = [reshape_dims(b) for b in biases]

    # DeepSpeed attn. kernel requires inputs to be type bf16 or fp16
    # Cast to bf16 so kernel can be used during inference
    orig_dtype = q.dtype
    if orig_dtype not in [torch.bfloat16, torch.float16]:
        o = DS4Sci_EvoformerAttention(q.to(dtype=torch.bfloat16),
                                      k.to(dtype=torch.bfloat16),
                                      v.to(dtype=torch.bfloat16),
                                      [b.to(dtype=torch.bfloat16) for b in biases])

        o = o.to(dtype=orig_dtype)
    else:
        o = DS4Sci_EvoformerAttention(q, k, v, biases)

    o = o.reshape(orig_shape)
    return o


def _attention(query: torch.Tensor,
               key: torch.Tensor,
               value: torch.Tensor,
               biases: Optional[List[torch.Tensor]] = None
               ) -> torch.Tensor:
    # [*, H, C_hidden, K]
    key = permute_final_dims(key, (1, 0))
    # [*, H, Q, K]
    a = torch.matmul(query, key)
    for b in biases:
        a += b

    a = F.softmax(a, dim=-1)
    # [*, H, Q, C_hidden]
    a = torch.matmul(a, value)

    return a


class Attention(nn.Module):
    """
    Standard multi-head attention using AlphaFold's default layer
    initialization. Allows multiple bias vectors.

    Args:
        c_q: Input dimension of query data
        c_k: Input dimension of key data
        c_v: Input dimension of value data
        c_hidden: Per-head hidden dimension
        num_heads: Number of attention heads
        gating: Whether the output should be gated using query data
    """

    def __init__(
            self,
            c_q: int,
            c_k: int,
            c_v: int,
            c_hidden: int,
            num_heads: int,
            gating: bool = True
    ):
        super(Attention, self).__init__()
        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.num_heads = num_heads
        self.gating = gating
        self.linear_q = Linear(self.c_q, self.c_hidden * self.num_heads, bias=False, init='glorot')
        self.linear_k = Linear(self.c_k, self.c_hidden * self.num_heads, bias=False, init='glorot')
        self.linear_v = Linear(self.c_v, self.c_hidden * self.num_heads, bias=False, init='glorot')
        self.linear_o = Linear(self.c_hidden * self.num_heads, self.c_q, init='final')
        self.linear_g = None
        if self.gating:
            self.linear_g = Linear(self.c_q, self.c_hidden * self.num_heads, init='gating')

    def project_qkv(self,
                    q_x: torch.Tensor,
                    kv_x: torch.Tensor = None
                    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        kv_x = q_x if kv_x is None else kv_x
        # [*, Q/K/V, H * c]
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)

        # [*, Q/K, num_heads, c]
        q = q.view(q.shape[:-1] + (self.num_heads, -1))
        k = k.view(k.shape[:-1] + (self.num_heads, -1))
        v = v.view(v.shape[:-1] + (self.num_heads, -1))

        return q, k, v

    def wrap_up(self,
                o: torch.Tensor,
                q_x: torch.Tensor
                ) -> torch.Tensor:
        if self.gating:
            g = self.linear_g(q_x).sigmoid()
            # [*, Q, H, C_hidden]
            g = g.view(g.shape[:-1] + (self.num_heads, -1))
            o = o * g

        # [*, Q, H * C_hidden]
        o = flatten_final_dims(o, 2)
        # [*, Q, C_q]
        o = self.linear_o(o)

        return o

    def forward(
            self,
            q_x: torch.Tensor,
            kv_x: torch.Tensor = None,
            biases: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Args:
            q_x: [*, Lq, Cq] query data
            kv_x: [*, Lk, Ck] key data
            biases: List of biases that broadcast to [*, H, Lq, Lk]

        Returns:
            [*, Lq, Cq] attention update
        """
        if biases is None:
            biases = []

        # [*, H, Q/K, C_hidden]
        q, k, v = self.project_qkv(q_x, kv_x)
        # when two more biases
        use_ds_attn = DS4Sci_EvoformerAttention is not None and len(biases) > 2
        if use_ds_attn:
            o = _deepspeed_evo_attn(q, k, v, biases)
        else:
            # [*, H, Q/K, C_hidden]
            q = q.transpose(-2, -3)
            k = k.transpose(-2, -3)
            v = v.transpose(-2, -3)
            q /= math.sqrt(self.c_hidden)
            o = _attention(q, k, v, biases=biases)
            o = o.transpose(-2, -3)

        o = self.wrap_up(o, q_x)

        return o


class GlobalAttention(nn.Module):
    def __init__(self, dim, hidden_dim, num_heads, inf, eps):
        super(GlobalAttention, self).__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.inf = inf
        self.eps = eps
        self.linear_q = Linear(dim, hidden_dim * num_heads, bias=False, init='glorot')
        self.linear_k = Linear(dim, hidden_dim, bias=False, init='glorot')
        self.linear_v = Linear(dim, hidden_dim, bias=False, init='glorot')
        self.linear_g = Linear(dim, hidden_dim * num_heads, init='gating')
        self.linear_o = Linear(hidden_dim * num_heads, dim, init='final')

    def forward(self, m: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # [*, N_res, C_in]
        q = torch.sum(m * mask.unsqueeze(-1), dim=-2) / (
                torch.sum(mask, dim=-1)[..., None] + self.eps
        )
        # [*, N_res, H * C_hidden]
        q = self.linear_q(q)
        q *= (self.hidden_dim ** (-0.5))
        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.num_heads, -1))
        # [*, N_res, N_seq, C_hidden]
        k = self.linear_k(m)
        v = self.linear_v(m)
        bias = (self.inf * (mask - 1))[..., :, None, :]
        # [*, N_res, H, N_seq]
        a = torch.matmul(q, k.transpose(-1, -2), )
        a += bias
        a = F.softmax(a, dim=-1)
        # [*, N_res, H, C_hidden]
        o = torch.matmul(a, v)
        # [*, N_res, N_seq, C_hidden]
        g = self.linear_g(m).sigmoid()
        # [*, N_res, N_seq, H, C_hidden]
        g = g.view(g.shape[:-1] + (self.num_heads, -1))
        # [*, N_res, N_seq, H, C_hidden]
        o = o.unsqueeze(-3) * g
        # [*, N_res, N_seq, H * C_hidden]
        o = o.reshape(o.shape[:-2] + (-1,))
        # [*, N_res, N_seq, C_in]
        m = self.linear_o(o)

        return m


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attention1 = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.attention2 = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, sfea_tns1, sfea_tns2):
        """
        Args:
            sfea_tns1: sequential feature_1 of size N x L1 x c_s
            sfea_tns2: sequential feature_2 of size N x L2 x c_s

        Returns:
            sfea_tns: merged sequential feature of size N x (L1+L2) x c_s
        """
        attn1, _ = self.attention1(sfea_tns1, sfea_tns2, sfea_tns2)
        attn2, _ = self.attention2(sfea_tns2, sfea_tns1, sfea_tns1)

        sfea_tns = torch.cat([attn1, attn2], dim=1)

        return sfea_tns

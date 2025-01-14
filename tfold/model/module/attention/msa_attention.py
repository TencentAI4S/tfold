# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
# Data: 2023/10/30 15:35
from functools import partial
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from tfold.model.layer import Linear, LayerNorm
from tfold.model.utils import chunk_layer
from tfold.utils.tensor import permute_final_dims
from .gating_multihead_attention import GatedMultiheadAttention


class GlobalAttention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 *,
                 pack_qkv: bool = False,
                 gating: bool = True,
                 inf=1e9,
                 eps=1e-10):
        super(GlobalAttention, self).__init__()
        self.dim = dim
        assert dim % num_heads == 0, f"dim({dim}) must be divisible by num_heads({num_heads})"
        self.head_dim = dim // num_heads
        self.num_heads = num_heads
        self.gating = gating
        self.inf = inf
        self.eps = eps
        self.pack_qkv = pack_qkv
        if self.pack_qkv:
            self.qkv_proj = Linear(dim, self.head_dim * num_heads + 2 * self.head_dim, bias=False, init="glorot")
        else:
            self.linear_q = Linear(dim, self.head_dim * num_heads, bias=False, init="glorot")
            self.linear_k = Linear(dim, self.head_dim, bias=False, init="glorot")
            self.linear_v = Linear(dim, self.head_dim, bias=False, init="glorot")

        if self.gating:
            self.linear_g = Linear(dim, self.head_dim * num_heads, init="gating")

        self.linear_o = Linear(self.head_dim * num_heads, dim, init="final")

    def _project_qkv(self, m, attn_mask=None):
        # masked mean(dim=-2) [*, N_res, C_in]
        if attn_mask is not None:
            q = torch.sum(m * attn_mask.unsqueeze(-1), dim=-2) / (torch.sum(attn_mask, dim=-1)[..., None] + self.eps)
        else:
            q = torch.mean(m, dim=-2)

        # [*, seq_len, dim]
        q = self.linear_q(q)
        # [*, seq_len, num_heads, head_dim]
        q = q.view(q.shape[:-1] + (self.num_heads, -1))

        # [*, seq_len, N_seq, head_dim]
        k = self.linear_k(m)
        v = self.linear_v(m)
        return q, k, v

    def forward(self, m: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            m: [*, seq_len, seq_len, dim]
            attn_mask: [*, seq_len, seq_len]

        Returns:
            output: [*, seq_len, seq_len, dim]
        """
        q, k, v = self._project_qkv(m, attn_mask)
        q *= self.head_dim ** (-0.5)
        # [*, seq_len, num_heads, N_seq]
        a = torch.matmul(q, k.transpose(-1, -2))
        if attn_mask is not None:
            bias = (self.inf * (attn_mask - 1))[..., :, None, :]
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


class MSAAttention(nn.Module):
    """msa axial attention

    Args:
        dim: Input channel dimension
        num_heads: Number of attention heads
        inf: A large number to be used in computing the attention mask
    """

    def __init__(self, dim, num_heads, head_dim=None, inf=1e9):
        super(MSAAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.inf = inf
        self.layer_norm_m = LayerNorm(self.dim)
        self.mha = GatedMultiheadAttention(self.dim,
                                           num_heads=self.num_heads,
                                           head_dim=head_dim)

    @torch.jit.ignore
    def _chunk(self,
               m: torch.Tensor,
               biases: Optional[List[torch.Tensor]],
               chunk_size: int,
               ) -> torch.Tensor:
        def fn(m, biases):
            m = self.layer_norm_m(m)
            return self.mha(m, biases=biases)

        inputs = {"m": m}
        if biases is not None:
            inputs["biases"] = biases
        else:
            fn = partial(fn, biases=None)

        return chunk_layer(
            fn,
            inputs,
            chunk_size=chunk_size,
            no_batch_dims=len(m.shape[:-2])
        )

    def forward(self,
                m: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                chunk_size: Optional[int] = None
                ) -> torch.Tensor:
        """
        Args:
            m: [*, N_seq, N_res, C_m] MSA embedding
            mask: [*, N_seq, N_res] MSA mask
            chunk_size: Size of chunks into which the inputs are split along their batch dimensions.
        """
        biases = []
        if mask is not None:
            # [*, N_seq, 1, 1, N_res]
            mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]
            biases.append(mask_bias)

        if chunk_size is not None:
            return self._chunk(
                m,
                biases,
                chunk_size
            )
        m = self.layer_norm_m(m)
        m = self.mha(m, biases=biases)
        return m


MSARowAttention = MSAAttention


class MSARowAttentionWithPairBias(MSAAttention):
    """
    Implements Algorithm 7.

    Args:
        c_m: Input channel dimension
        c_z: Pair embedding channel dimension
        num_heads: Number of attention heads
        inf: Large number used to construct attention masks
    """

    def __init__(self, c_m, c_z, num_heads, inf=1e9):
        super(MSARowAttentionWithPairBias, self).__init__(c_m, num_heads, inf=inf)
        self.c_z = c_z
        self.layer_norm_z = LayerNorm(self.c_z)
        self.linear_z = Linear(self.c_z, self.num_heads, bias=False, init="normal")

    def forward(self,
                m: torch.Tensor,
                z: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                chunk_size: Optional[int] = None
                ) -> torch.Tensor:
        """
        Args:
            m: [*, N_seq, N_res, C_m] MSA embedding
            z: [*, N_res, N_res, C_z] pair embedding. Required only if pair_bias is True
            mask: [*, N_seq, N_res] MSA mask
            chunk_size: Size of chunks into which the inputs are split along their batch dimensions.
        """
        z = self.layer_norm_z(z)
        z = self.linear_z(z)
        # [*, 1, num_heads, N_res, N_res]
        z = permute_final_dims(z, (2, 0, 1)).unsqueeze(-4)
        biases = [z, ]

        if mask is not None:
            # [*, N_seq, 1, 1, N_res]
            mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]
            biases.append(mask_bias)

        if chunk_size is not None:
            return self._chunk(
                m,
                biases,
                chunk_size
            )

        m = self.layer_norm_m(m)
        m = self.mha(m, biases=biases)
        return m


class MSAColumnAttention(nn.Module):

    def __init__(self, c_m, num_heads, head_dim=None, inf=1e9):
        """
        Args:
            c_m: MSA channel dimension
            c_hidden: Per-head hidden channel dimension
            no_heads: Number of attention heads
            inf: Large number used to construct attention masks
        """
        super(MSAColumnAttention, self).__init__()
        self.c_m = c_m
        self.inf = inf
        self._msa_att = MSAAttention(c_m,
                                     num_heads=num_heads,
                                     head_dim=head_dim,
                                     inf=inf)

    def forward(self,
                m: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                chunk_size: Optional[int] = None
                ) -> torch.Tensor:
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
            mask:
                [*, N_seq, N_res] MSA mask
            chunk_size:
                Size of chunks into which the inputs are split along their
                batch dimensions. A low value decreases memory overhead at the 
                cost of slower execution. Chunking is not performed by default.
        """
        # [*, N_res, N_seq, C_in]
        m = m.transpose(-2, -3)
        if mask is not None:
            mask = mask.transpose(-1, -2)

        m = self._msa_att(m, mask=mask, chunk_size=chunk_size)

        # [*, N_seq, N_res, C_in]
        m = m.transpose(-2, -3)
        return m


class MSAColumnGlobalAttention(nn.Module):
    def __init__(self, dim, num_heads, inf=1e9, eps=1e-10):
        super(MSAColumnGlobalAttention, self).__init__()
        self.dim = dim
        assert self.dim % num_heads == 0, f"dim({dim}) must be divisible by num_heads({num_heads})"
        self.head_dim = dim // num_heads
        self.num_heads = num_heads
        self.inf = inf
        self.eps = eps
        self.layer_norm_m = LayerNorm(dim)
        self.global_attention = GlobalAttention(
            dim=dim,
            num_heads=num_heads,
            inf=inf,
            eps=eps
        )

    @torch.jit.ignore
    def _chunk(self,
               m: torch.Tensor,
               mask: torch.Tensor,
               chunk_size: int
               ) -> torch.Tensor:
        mha_input = {
            "m": m,
            "mask": mask,
        }

        def fn(m, mask):
            m = self.layer_norm_m(m)
            return self.global_attention(m, mask)

        return chunk_layer(
            fn,
            mha_input,
            chunk_size=chunk_size,
            no_batch_dims=len(m.shape[:-2]),
        )

    def forward(
            self,
            m: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            chunk_size: Optional[int] = None
    ) -> torch.Tensor:
        # [*, N_res, N_seq, C_in]
        m = m.transpose(-2, -3)
        if mask is not None:
            mask = mask.transpose(-1, -2)

        if chunk_size is not None:
            m = self._chunk(m, mask, chunk_size)
        else:
            m = self.layer_norm_m(m)
            m = self.global_attention(m=m, mask=mask)

        # [*, N_seq, N_res, C_in]
        m = m.transpose(-2, -3)

        return m

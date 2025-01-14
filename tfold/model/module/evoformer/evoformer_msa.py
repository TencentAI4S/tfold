# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
# Data: 2023/10/30 15:35
from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn as nn

from tfold.model.layer import Linear, DropoutRowwise, LayerNorm
from tfold.model.utils import checkpoint_blocks, chunk_layer
from .outer_product_mean import OuterProductMeanMSA
from ..attention import (TriangleAttention, MSARowAttentionWithPairBias, MSAColumnAttention,
                         TriangleMultiplicationIncoming, TriangleMultiplicationOutgoing)


class Transition(nn.Module):
    """
    Implements Algorithm 9 and 15. PairTransition and MSATransition
    """

    def __init__(self, c_in, n=4):
        """
        Args:
            c: Transition channel dimension
        """
        super(Transition, self).__init__()
        self.c_in = c_in
        self.c_hidden = n * c_in
        self.layer_norm = LayerNorm(self.c_in)
        self.linear_1 = Linear(self.c_in, self.c_hidden, init="relu")
        self.relu = nn.ReLU()
        self.linear_2 = Linear(self.c_hidden, self.c_in, init="final")

    @torch.jit.ignore
    def _chunk(self,
               x: torch.Tensor,
               mask: torch.Tensor,
               chunk_size: int,
               ) -> torch.Tensor:
        inputs = {"x": x}
        if mask is not None:
            inputs["mask"] = mask

        return chunk_layer(
            self._transition,
            inputs,
            chunk_size=chunk_size,
            no_batch_dims=len(x.shape[:-2]),
        )

    def _transition(self, x, mask=None):
        x = self.layer_norm(x)
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        if mask is not None:
            x = x * mask

        return x

    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                chunk_size: Optional[int] = None,
                ) -> torch.Tensor:
        """
        Args:
            x: [*, c] embedding
            mask: [*,] mask

        Returns:
            [*, c] embedding update
        """
        if mask is not None:
            mask = mask[..., None]

        if chunk_size is not None:
            x = self._chunk(x, mask, chunk_size=chunk_size)
        else:
            x = self._transition(x=x, mask=mask)

        return x


class EvoformerBlockCore(nn.Module):
    def __init__(
            self,
            c_m: int,
            c_z: int,
            c_hidden_opm: int,
            c_hidden_mul: int,
            num_heads: int,
            transition_n: int = 4,
            pair_dropout: float = 0.0,
            inf: float = 1e9
    ):
        super(EvoformerBlockCore, self).__init__()
        self.msa_transition = Transition(c_m, n=transition_n)
        self.outer_product_mean = OuterProductMeanMSA(c_m, c_z, c_hidden_opm)
        self.tri_mul_out = TriangleMultiplicationOutgoing(c_z, c_hidden_mul)
        self.tri_mul_in = TriangleMultiplicationIncoming(c_z, c_hidden_mul)
        self.tri_att_start = TriangleAttention(c_z, num_heads, inf=inf)
        self.tri_att_end = TriangleAttention(c_z, num_heads, inf=inf)
        self.pair_transition = Transition(c_z)
        self.dropout = DropoutRowwise(pair_dropout)

    def forward(self,
                m: torch.Tensor,
                z: torch.Tensor,
                msa_mask: Optional[torch.Tensor] = None,
                pair_mask: Optional[torch.Tensor] = None,
                chunk_size: Optional[int] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        m = m + self.msa_transition(m, chunk_size=chunk_size)
        z = z + self.outer_product_mean(m, mask=msa_mask, chunk_size=chunk_size)

        z = z + self.dropout(self.tri_mul_out(z, mask=pair_mask))
        z = z + self.dropout(self.tri_mul_in(z, mask=pair_mask))
        z = z + self.dropout(self.tri_att_start(z, mask=pair_mask, chunk_size=chunk_size))

        z = z.transpose(-2, -3)
        z = z + self.dropout(
            self.tri_att_end(z,
                             mask=pair_mask.transpose(-1, -2) if pair_mask is not None else None,
                             chunk_size=chunk_size)
        )
        z = z.transpose(-2, -3)

        z = z + self.pair_transition(z, chunk_size=chunk_size)

        return m, z


class EvoformerBlock(nn.Module):

    def __init__(self,
                 c_m: int,
                 c_z: int,
                 c_hidden_opm: int,
                 c_hidden_mul: int,
                 num_heads_msa: int,
                 num_heads_pair: int,
                 use_column_attention: bool = True,
                 msa_dropout: float = 0.0,
                 pair_dropout: float = 0.0,
                 inf: float = 1e9):
        super(EvoformerBlock, self).__init__()
        self.msa_att_row = MSARowAttentionWithPairBias(
            c_m=c_m,
            c_z=c_z,
            num_heads=num_heads_msa,
            inf=inf
        )
        self.use_column_attention = use_column_attention
        if self.use_column_attention:
            self.msa_att_col = MSAColumnAttention(
                c_m,
                num_heads=num_heads_msa,
                inf=inf
            )
        self.msa_dropout = DropoutRowwise(msa_dropout)
        self.core = EvoformerBlockCore(
            c_m=c_m,
            c_z=c_z,
            c_hidden_opm=c_hidden_opm,
            c_hidden_mul=c_hidden_mul,
            num_heads=num_heads_pair,
            pair_dropout=pair_dropout,
            inf=inf
        )

    def forward(self,
                m: Optional[torch.Tensor],
                z: Optional[torch.Tensor],
                msa_mask: Optional[torch.Tensor] = None,
                pair_mask: Optional[torch.Tensor] = None,
                chunk_size: Optional[int] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        # row-wise attention
        m = m + self.msa_dropout(
            self.msa_att_row(
                m,
                z=z,
                mask=msa_mask,
                chunk_size=chunk_size
            )
        )
        # col-wise attention
        if self.use_column_attention:
            m = m + self.msa_att_col(
                m,
                mask=msa_mask,
                chunk_size=chunk_size)

        m, z = self.core(m, z,
                         msa_mask=msa_mask,
                         pair_mask=pair_mask,
                         chunk_size=chunk_size)

        return m, z


class EvoformerStack(nn.Module):
    """Main Evoformer trunk.

    Args:
        c_m: MSA channel dimension
        c_s:  Channel dimension of the output "single" embedding
        c_z: Pair channel dimension
        c_hidden_opm: Hidden dimension in outer product mean module
        c_hidden_mul: Hidden dimension in multiplicative updates
        num_heads_msa:
            Number of heads used for MSA attention
        num_heads_pair:
            Number of heads used for pair attention
        num_blocks:
            Number of Evoformer blocks in the stack
        msa_dropout:
            Dropout rate for MSA activations
        pair_dropout:
            Dropout used for pair activations
    """

    def __init__(
            self,
            c_m: int = 256,
            c_s: int = 384,
            c_z: int = 128,
            c_hidden_opm: int = 32,
            c_hidden_mul: int = 128,
            no_heads_msa: int = 8,
            no_heads_pair: int = 4,
            no_blocks: int = 48,
            use_column_attention: bool = True,
            msa_dropout: float = 0.0,
            pair_dropout: float = 0.0,
            **kwargs
    ):
        super(EvoformerStack, self).__init__()
        self.blocks = nn.ModuleList()
        for _ in range(no_blocks):
            block = EvoformerBlock(
                c_m=c_m,
                c_z=c_z,
                c_hidden_opm=c_hidden_opm,
                c_hidden_mul=c_hidden_mul,
                num_heads_msa=no_heads_msa,
                num_heads_pair=no_heads_pair,
                use_column_attention=use_column_attention,
                msa_dropout=msa_dropout,
                pair_dropout=pair_dropout
            )
            self.blocks.append(block)

        self.linear = Linear(c_m, c_s)
        self.chunk_size = None
        self.checkpoint_interval = None

    def set_chunk_size(self, chunk_size):
        self.chunk_size = chunk_size

    def enable_activation_checkpoint(self, enabled=True, interval=1):
        if enabled:
            self.checkpoint_interval = interval
        else:
            self.checkpoint_interval = None

    def forward(self,
                m: torch.Tensor,
                z: torch.Tensor,
                msa_mask: torch.Tensor = None,
                pair_mask: torch.Tensor = None,
                chunk_size: int = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            m: [*, N_seq, N_res, C_m] MSA embedding
            z: [*, N_res, N_res, C_z] pair embedding
            msa_mask: [*, N_seq, N_res] MSA mask
            pair_mask:  [*, N_res, N_res] pair mask

        Returns:
            m: [*, N_seq, N_res, C_m] MSA embedding
            z: [*, N_res, N_res, C_z] pair embedding
            s: [*, N_res, C_s] single embedding (or None if extra MSA stack)
        """
        blocks = [
            partial(
                b,
                msa_mask=msa_mask,
                pair_mask=pair_mask,
                chunk_size=chunk_size
            )
            for b in self.blocks
        ]
        m, z = checkpoint_blocks(
            blocks,
            args=(m, z),
            interval=self.checkpoint_interval
        )
        s = m[..., 0, :, :]
        s = self.linear(s)

        return m, z, s

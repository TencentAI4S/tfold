# Copyright (c) 2024, Tencent Inc. All rights reserved.
# Data: 2023/12/29 14:37
from functools import partial
from typing import Tuple, Optional

import torch
import torch.nn as nn

from tfold.model.layer import DropoutRowwise
from tfold.model.module.attention import MSAColumnGlobalAttention, MSARowAttentionWithPairBias
from tfold.model.module.evoformer import EvoformerBlockCore


class ExtraMSABlock(nn.Module):
    """
        Almost identical to the standard EvoformerBlock, except in that the
        ExtraMSABlock uses GlobalAttention for MSA column attention and
        requires more fine-grained control over checkpointing. Separated from
        its twin to preserve the TorchScript-ability of the latter.
    """

    def __init__(self,
                 c_m: int,
                 c_z: int,
                 c_hidden_opm: int,
                 c_hidden_mul: int,
                 no_heads_msa: int,
                 no_heads_pair: int,
                 msa_dropout: float,
                 pair_dropout: float,
                 inf: float,
                 eps: float
                 ):
        super(ExtraMSABlock, self).__init__()
        self.msa_att_row = MSARowAttentionWithPairBias(
            c_m=c_m,
            c_z=c_z,
            num_heads=no_heads_msa,
            inf=inf
        )
        self.msa_att_col = MSAColumnGlobalAttention(
            dim=c_m,
            num_heads=no_heads_msa,
            inf=inf,
            eps=eps,
        )
        self.msa_dropout_layer = DropoutRowwise(msa_dropout)
        self.core = EvoformerBlockCore(
            c_m=c_m,
            c_z=c_z,
            c_hidden_opm=c_hidden_opm,
            c_hidden_mul=c_hidden_mul,
            num_heads=no_heads_pair,
            pair_dropout=pair_dropout,
            inf=inf
        )

    def forward(self,
                m: Optional[torch.Tensor],
                z: Optional[torch.Tensor],
                msa_mask: torch.Tensor = None,
                pair_mask: torch.Tensor = None,
                chunk_size: Optional[int] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        m = m + self.msa_dropout_layer(
            self.msa_att_row(
                m.clone() if torch.is_grad_enabled() else m,
                z=z.clone() if torch.is_grad_enabled() else z,
                mask=msa_mask,
                chunk_size=chunk_size
            )
        )

        m = m + self.msa_att_col(m, mask=msa_mask, chunk_size=chunk_size)
        m, z = self.core(m, z,
                         msa_mask=msa_mask,
                         pair_mask=pair_mask,
                         chunk_size=chunk_size)

        return m, z


class ExtraMSAStack(nn.Module):
    """
    Implements Algorithm 18.
    """

    def __init__(self,
                 c_m: int = 64,
                 c_z: int = 128,
                 c_hidden_opm: int = 128,
                 c_hidden_mul: int = 32,
                 no_heads_msa: int = 8,
                 no_heads_pair: int = 4,
                 no_blocks: int = 4,
                 msa_dropout: float = 0.15,
                 pair_dropout: float = 0.25,
                 inf: float = 1e-9,
                 eps: float = 1e-8,
                 **kwargs
                 ):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(no_blocks):
            block = ExtraMSABlock(
                c_m=c_m,
                c_z=c_z,
                c_hidden_opm=c_hidden_opm,
                c_hidden_mul=c_hidden_mul,
                no_heads_msa=no_heads_msa,
                no_heads_pair=no_heads_pair,
                msa_dropout=msa_dropout,
                pair_dropout=pair_dropout,
                inf=inf,
                eps=eps
            )
            self.blocks.append(block)

    def forward(self,
                m: torch.Tensor,
                z: torch.Tensor,
                msa_mask: Optional[torch.Tensor] = None,
                pair_mask: Optional[torch.Tensor] = None,
                chunk_size: int = None,
                ) -> torch.Tensor:
        """
        Args:
            m:
                [*, N_extra, N_res, C_m] extra MSA embedding
            z:
                [*, N_res, N_res, C_z] pair embedding
            chunk_size: Inference-time subbatch size for Evoformer modules
            use_lma: Whether to use low-memory attention during inference
            msa_mask:
                Optional [*, N_extra, N_res] MSA mask
            pair_mask:
                Optional [*, N_res, N_res] pair mask
        Returns:
            [*, N_res, N_res, C_z] pair update
        """
        blocks = [
            partial(
                b,
                msa_mask=msa_mask,
                pair_mask=pair_mask,
                chunk_size=chunk_size,
            ) for b in self.blocks
        ]

        for b in blocks:
            m, z = b(m, z)

        return z

# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
# Data: 2023/12/22 11:16
import torch
import torch.nn as nn
import torch.nn.functional as F

from tfold.model.layer import Linear
from .extra_msa_stack import ExtraMSAStack


class ExtraMSAEmbedder(nn.Module):
    """
    Embeds unclustered MSA sequences.

    Implements Algorithm 2, line 15
    """

    def __init__(
            self,
            c_in: int,
            c_out: int
    ):
        """
        Args:
            c_in: Input channel dimension
            c_out: Output channel dimension
        """
        super(ExtraMSAEmbedder, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.linear = Linear(self.c_in, self.c_out)

    @property
    def dtype(self):
        return self.linear.weight.dtype

    def forward(self,
                extra_msa,
                extra_has_deletion,
                extra_deletion_value,
                ) -> torch.Tensor:
        """
        Args:
            msa_feat: [*, N_extra_seq, N_res, C_in] "extra_msa_feat" features

        Returns:
            [*, N_extra_seq, N_res, C_out] embedding
        """
        msa_1hot = F.one_hot(extra_msa, 23)
        msa_feat = [
            msa_1hot,
            extra_has_deletion.unsqueeze(-1),
            extra_deletion_value.unsqueeze(-1),
        ]
        msa_feat = torch.cat(msa_feat, dim=-1)
        if msa_feat.dtype != self.dtype:
            msa_feat = msa_feat.to(self.dtype)

        x = self.linear(msa_feat)

        return x


class ExtraMSAEmbedding(nn.Module):

    def __init__(self, config):
        super(ExtraMSAEmbedding, self).__init__()
        self.msa_embedding = ExtraMSAEmbedder(**config.extra_msa_embedder)
        self.msa_stack = ExtraMSAStack(**config.extra_msa_stack)

    def forward(self, batch, z, pair_mask, chunk_size=None):
        msa_mask = batch.get("extra_msa_mask", None)
        if msa_mask is not None:
            msa_mask = msa_mask.to(dtype=z.dtype)

        # [*, S_e, N, C_e]
        a = self.msa_embedding(
            batch["extra_msa"],
            batch["extra_has_deletion"],
            batch["extra_deletion_value"]
        )
        # [*, N, N, C_z]
        z = self.msa_stack(
            a, z,
            msa_mask=msa_mask,
            pair_mask=pair_mask,
            chunk_size=chunk_size,
        )
        return z

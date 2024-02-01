# -*- coding: utf-8 -*-
# Copyright (c) 2023, Tencent Inc. All rights reserved.
from torch import nn

from tfold.protein import prot_constants as pc
from .onehot_embedding import OnehotEmbedding


class LearnableResidueEmbedding(nn.Module):
    """Learnable amino-acid residue encoder."""

    def __init__(self, c_s, c_z):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.onehot = OnehotEmbedding(pc.RESD_NAMES_1C)
        self.embed = nn.Embedding(len(pc.RESD_NAMES_1C), self.c_s)
        self.linear_left = nn.Linear(self.c_s, self.c_z)
        self.linear_right = nn.Linear(self.c_s, self.c_z)

    def forward(self, aa_seq):
        """Get initial amino-acid residue encodings.

        Args:
            aa_seq: amino-acid sequence of length L

        Returns:
            s: initial single features of size N x L x c_s
            z: initial pair features of size N x L x L x c_z
        """
        device = self.embed.weight.device
        # get initial single features
        idxs_vec = self.onehot.name2idx(aa_seq).to(device)
        s = self.embed(idxs_vec).unsqueeze(dim=0)
        # get initial pair features
        sfea_tns_left = self.linear_left(s)
        sfea_tns_right = self.linear_right(s)
        z = sfea_tns_left[:, :, None] + sfea_tns_right[:, None]

        return s, z

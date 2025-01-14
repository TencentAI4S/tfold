# Copyright (c) 2024, Tencent Inc. All rights reserved.
# Data: 2024/5/29 19:58
from typing import Tuple

import torch

from tfold.model.layer import Linear
from .relative_position_embedding import RelativePositionEmbedding


class InputEmbedding(RelativePositionEmbedding):
    """
    Embeds a subset of the input features.
    Implements Algorithms 3 (InputEmbedder) and 4 (relpos).

    Args:
        tf_dim: Final dimension of the target features
        msa_dim: Final dimension of the MSA features
        c_z: Pair embedding dimension
        c_m: MSA embedding dimension
        relpos_k: Window size used in relative positional encoding
    """

    def __init__(
            self,
            c_m: int = 256,
            c_z: int = 128,
            tf_dim: int = 22,
            msa_dim: int = 49,
            relpos_k: int = 32,
    ):
        super().__init__(c_m, c_z, tf_dim, relpos_k=relpos_k)
        self.msa_dim = msa_dim
        self.linear_msa_m = Linear(msa_dim, c_m)
    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def forward(self, msa_feat, target_feat, residue_index=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            msa_feat: [*, N_clust, seq_len, msa_dim], msa features
            target_feat: [*, seq_len, tf_dim], target features
            ri: [*, seq_len], residue index

        Returns:
            msa_emb: [*, N_clust, seq_len, C_m] MSA embedding
            pair_emb: [*, seq_len, seq_len, C_z] pair embedding
        """
        if msa_feat.dtype != self.dtype:
            msa_feat = msa_feat.to(self.dtype)
            target_feat = target_feat.to(self.dtype)

        m, z = super().forward(target_feat, residue_index=residue_index)
        # [*, num_seqs, seq_len, c_m]
        num_seqs = msa_feat.shape[-3]
        m = m.expand(((-1,) * len(msa_feat.shape[:-3]) + (num_seqs, -1, -1)))
        m = self.linear_msa_m(msa_feat) + m

        return m, z

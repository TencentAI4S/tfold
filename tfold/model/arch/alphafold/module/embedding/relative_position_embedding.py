# Copyright (c) 2024, Tencent Inc. All rights reserved.
# Data: 2024/5/29 19:54
from typing import Tuple

import torch
from torch import nn

from tfold.model.layer import Linear


class RelativePositionEmbedding(nn.Module):
    """
    Embeds the sequence pre-embedding passed to the model and the target_feat features.
    Args:
        c_z: Pair embedding dimension
        c_m: Single-Seq embedding dimension
        seq_dim: End channel dimension of the incoming target features
        relpos_k: Window size used in relative position encoding
    """

    def __init__(self,
                 c_m: int,
                 c_z: int,
                 seq_dim: int = 22,
                 relpos_k: int = 32):
        super(RelativePositionEmbedding, self).__init__()
        self.seq_dim = seq_dim
        self.c_z = c_z
        self.c_m = c_m
        self.linear_tf_z_i = Linear(seq_dim, c_z)
        self.linear_tf_z_j = Linear(seq_dim, c_z)
        self.linear_tf_m = Linear(seq_dim, c_m)
        self.relpos_k = relpos_k
        self.num_bins = 2 * relpos_k + 1
        self.linear_relpos = Linear(self.num_bins, c_z)

    def relative_position_embedding(self, ri: torch.Tensor):
        """Computes relative positional encodings

        Args:
            ri: [*, N], residue_index

        Returns:
            z: [*, N, N, c_z], embedding
        """
        d = ri[..., None] - ri[..., None, :]
        boundaries = torch.arange(
            start=-self.relpos_k, end=self.relpos_k + 1, device=d.device
        )
        reshaped_bins = boundaries.view(((1,) * len(d.shape)) + (len(boundaries),))
        d = d[..., None] - reshaped_bins
        d = torch.abs(d)
        d = torch.argmin(d, dim=-1)  # [*, seq_len, seq_len]
        d = nn.functional.one_hot(d, num_classes=len(boundaries)).to(ri.dtype)
        return self.linear_relpos(d)

    def forward(self, target_feat, residue_index=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            sf: [*, seq_len, tf_dim], sequence features
            ri: [*, seq_len], residue index

        Returns:
            msa_emb: [*, num_seqs, seq_len, C_m] MSA embedding
            pair_emb: [*, num_seqs, seq_len, C_z] pair embedding
        """
        sf = target_feat
        ri = residue_index
        if ri is None:
            bs, seq_len = sf.shape[:-1]
            ri = torch.arange(seq_len, device=sf.device)[None]
            ri = ri.expand((bs, seq_len))

        ri = ri.type(sf.dtype)
        # [*, N_res, c_z]
        sf_emb_i = self.linear_tf_z_i(sf)
        sf_emb_j = self.linear_tf_z_j(sf)
        m = self.linear_tf_m(sf).unsqueeze(-3)
        z = self.relative_position_embedding(ri)
        z = z + (sf_emb_i[..., None, :] + sf_emb_j[..., None, :, :])

        return m, z

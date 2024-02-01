# -*- coding: utf-8 -*-
# Copyright (c) 2023, Tencent Inc. All rights reserved.
import torch
import torch.nn as nn

from .position_embedding import SinusoidalPositionEmbedding


class MultimerPositionEmebedding(nn.Module):
    """Multimer positional encoder.

    Args:
        dim: number of dimensions in positional encodings
        max_len: maximal number of residues per chain
    Notes:
    * This encoder adopts shared sinusoidal positional encoder for all the chains. Therefore, chain
        relative positional encoder must be used for multimer structure prediction. Otherwise, the
        model will not be able to distinguish positional encodings for different chains.
    * This module does not contain any learnable parameters and cannot determine which computational
        device should be used. Thus, the resulting positional encodings should be manually moved to
        the correct device.
    """

    def __init__(self, dim, max_len=256):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        # build a shared sinusoidal positional encoder for all the chain types
        self.posi_encoder = SinusoidalPositionEmbedding(self.dim, self.max_len)

    def forward(self, chn_infos):
        """Get multimer positional encodings.

        Args:
            chn_infos: list of (chain_id, n_resds) tuples

        Returns:
            penc_mat: positional encodings of size L x D
        """

        # get multimer positional encodings
        idxs_vec = torch.cat([torch.arange(n_resds) for _, n_resds in chn_infos], dim=0)
        penc_mat = self.posi_encoder(idxs_vec)

        return penc_mat

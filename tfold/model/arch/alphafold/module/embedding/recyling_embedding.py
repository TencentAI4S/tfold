# Copyright (c) 2024, Tencent Inc. All rights reserved.
# Data: 2024/5/29 19:54
from typing import Tuple

import torch
import torch.nn as nn

from tfold.model.layer import Linear, LayerNorm


class RecyclingEmbedding(nn.Module):
    """
    Embeds the output of an iteration of the model for recycling.
    Implements Algorithm 32.

    Args:
        c_m: MSA channel dimension
        c_z: Pair embedding channel dimension
        min_bin: Smallest distogram bin (Angstroms)
        max_bin: Largest distogram bin (Angstroms)
        num_bins: Number of distogram bins
    """

    def __init__(
            self,
            c_m: int = 256,
            c_z: int = 128,
            min_bin: float = 3.25,
            max_bin: float = 20.75,
            no_bins: int = 15,
            inf: float = 1e8
    ):
        super(RecyclingEmbedding, self).__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.num_bins = no_bins
        self.inf = inf
        self.linear = Linear(self.num_bins, self.c_z)
        self.layer_norm_m = LayerNorm(self.c_m)
        self.layer_norm_z = LayerNorm(self.c_z)
        self.squared_bins = torch.linspace(
            self.min_bin,
            self.max_bin,
            self.num_bins
        ) ** 2

    def forward(
            self,
            m: torch.Tensor,
            z: torch.Tensor,
            x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            m: [*, N_res, C_m], First row of the MSA embedding.
            z: [*, N_res, N_res, C_z] pair embedding
            x: [*, N_res, 3] predicted C_beta coordinates

        Returns:
            m: [*, N_res, C_m] MSA embedding update
            z: [*, N_res, N_res, C_z] pair embedding update
        """
        m = self.layer_norm_m(m)
        z = self.layer_norm_z(z)
        # This squared method might become problematic in FP16 mode.
        squared_bins = self.squared_bins.to(m.device)
        upper = torch.cat(
            [squared_bins[1:], squared_bins.new_tensor([self.inf])], dim=-1
        )
        d = torch.sum(
            (x[..., None, :] - x[..., None, :, :]) ** 2, dim=-1, keepdims=True
        )
        # [*, N, N, no_bins]
        d = ((d > squared_bins) * (d < upper)).type(x.dtype)
        # [*, N, N, C_z]
        d = self.linear(d)
        z = z + d

        return m, z

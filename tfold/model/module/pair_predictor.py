# -*- coding: utf-8 -*-
# Copyright (c) 2023, Tencent Inc. All rights reserved.
import torch.nn as nn

from .mlp import MLP, LayerNorm
from ..layer import Linear


class PairPredictor(nn.Module):
    """predict distance map from pair features"""

    def __init__(self,
                 c_z,
                 bins=(37, 25, 25, 25),
                 is_use_ln=True,
                 pb_relax=False,
                 p_drop=0.0,
                 **kwargs):
        super(PairPredictor, self).__init__()
        self.c_z = c_z
        self.norm = LayerNorm(c_z, pb_relax=pb_relax) if is_use_ln else nn.Identity()
        self.proj = Linear(c_z, c_z)
        self.drop = nn.Dropout(p_drop)
        self.resnet_dist = MLP(dim=c_z, ffn_dim=c_z * 4, d_model_out=bins[0], p_drop=p_drop, **kwargs)
        self.resnet_omega = MLP(dim=c_z, ffn_dim=c_z * 4, d_model_out=bins[1], p_drop=p_drop,
                                **kwargs)
        self.resnet_theta = MLP(dim=c_z, ffn_dim=c_z * 4, d_model_out=bins[2], p_drop=p_drop,
                                **kwargs)
        self.resnet_phi = MLP(dim=c_z, ffn_dim=c_z * 4, d_model_out=bins[3], p_drop=p_drop,
                              **kwargs)

    def forward(self, z):
        # input: pair info (B, L, L, C)
        z = self.norm(z)
        z = self.drop(self.proj(z))
        # predict theta, phi (non-symmetric)
        logits_theta = self.resnet_theta(z).permute(0, 3, 1, 2)
        logits_phi = self.resnet_phi(z).permute(0, 3, 1, 2)
        # predict dist, omega
        z = 0.5 * (z + z.permute(0, 2, 1, 3))
        logits_dist = self.resnet_dist(z).permute(0, 3, 1, 2)
        logits_omega = self.resnet_omega(z).permute(0, 3, 1, 2)

        return logits_dist, logits_omega, logits_theta, logits_phi

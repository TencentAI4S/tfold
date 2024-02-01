# -*- coding: utf-8 -*-
# Copyright (c) 2023, Tencent Inc. All rights reserved.
import torch.nn as nn


class AngleHead(nn.Module):
    """The torsion angle prediction network."""

    def __init__(self, n_dims_sfcd, n_dims_hidd, num_angles=7):
        super().__init__()
        self.n_dims_sfcd = n_dims_sfcd
        self.n_dims_hidd = n_dims_hidd
        self.n_dims_angl = 2  # cosine & sine values for each torsion angle
        self.num_angles = num_angles
        self.net = nn.ModuleDict()
        self.net['linear-s'] = nn.Linear(self.n_dims_sfcd, self.n_dims_hidd)
        self.net['linear-i'] = nn.Linear(self.n_dims_sfcd, self.n_dims_hidd)
        self.net['mlp-1'] = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.n_dims_hidd, self.n_dims_hidd),
            nn.ReLU(),
            nn.Linear(self.n_dims_hidd, self.n_dims_hidd),
        )
        self.net['mlp-2'] = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.n_dims_hidd, self.n_dims_hidd),
            nn.ReLU(),
            nn.Linear(self.n_dims_hidd, self.n_dims_hidd),
        )
        self.net['mlp-3'] = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.n_dims_hidd, num_angles * self.n_dims_angl),
        )

    def forward(self, sfcd_tns, sfcd_tns_init):
        """Perform the forward pass.

        Args:
            sfea_tns: single features of size N x L x D_s
            sfea_tns_init: initial single features of size N x L x D_s

        Returns:
            angl_tns: torsion angle matrices of size N x L x K x 2
        """
        n_smpls, n_resds, _ = sfcd_tns.shape

        # obtain hidden representation for torsion angle predictions
        hfea_tns = self.net['linear-s'](sfcd_tns) + self.net['linear-i'](sfcd_tns_init)
        hfea_tns = hfea_tns + self.net['mlp-1'](hfea_tns)
        hfea_tns = hfea_tns + self.net['mlp-2'](hfea_tns)

        # predict torsion angles - deterministic
        angl_tns = self.net['mlp-3'](hfea_tns).view(n_smpls, n_resds, self.num_angles, -1)

        return angl_tns

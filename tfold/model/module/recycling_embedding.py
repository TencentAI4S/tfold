# -*- coding: utf-8 -*-
# Copyright (c) 2023, Tencent Inc. All rights reserved.
import torch
from torch import nn

from tfold.protein import ProtStruct
from tfold.protein.prot_constants import N_ATOMS_PER_RESD
from tfold.utils import cdist
from ..layer import Linear, LayerNorm


def get_cb_cords(aa_seq, cord_tns_all):
    """Get 3D coordinates of CB (CA for Glycine) atoms.

    Args:
    * aa_seq: amino-acid sequence
    * cord_tns_all: per-atom 3D coordinates of size L x M x 3

    Returns:
    * cord_mat_sel: CB (CA for Glycine) atoms' 3D coordinates of size L x 3
    """
    device = cord_tns_all.device
    # get 3D coordinates of CB (CA for Glycine) atoms
    atom_names = ['CA', 'CB']
    cmsk_mat = torch.tensor(
        [[1, 0] if x == 'G' else [0, 1] for x in aa_seq], dtype=torch.int8, device=device)
    cord_tns_raw = ProtStruct.get_atoms(aa_seq, cord_tns_all, atom_names)  # L x 2 x 3
    cord_mat_sel = torch.sum(cmsk_mat.unsqueeze(dim=2) * cord_tns_raw, dim=1)  # L x 3

    return cord_mat_sel


class RecyclingEmbedding(nn.Module):
    """The embedding network of <XFoldNet> & <AF2SModNet> modules' outputs for recycling.

    Args:
        c_m: number of dimensions in MSA features
        c_z: number of dimensions in pair features
    """

    def __init__(
            self,
            c_m=384,
            c_z=256,
            num_bins=18,
            dist_min=3.375,
            dist_max=21.375
    ):
        super().__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.num_bins = num_bins
        self.dist_min = dist_min
        self.dist_max = dist_max
        self.bin_width = (self.dist_max - self.dist_min) / self.num_bins
        self.norm_m = LayerNorm(self.c_m)
        self.norm_p = LayerNorm(self.c_z)
        self.linear = Linear(self.num_bins, self.c_z)

    def forward(self, aa_seq, m, z, rc_inputs=None):
        """
        Args:
            aa_seq: amino-acid sequence
            m: MSA features of size N x K x L x D_m
            z: pair features of size N x L x L x D_p
            rc_inputs: (optional) dict of additional inputs for recycling embeddings
                sfea: single features of size N x L x D_m
                pfea: pair features of size N x L x L x D_p
                cord: per-atom 3D coordinates of size L x M x 3

        Returns:
            m: updated MSA features of size N x K x L x D_m
            z: updated pair features of size N x L x L x D_p
        """
        seq_len = len(aa_seq)
        dtype = m.dtype  # for compatibility w/ half-precision inputs
        device = m.device
        # initialize additional inputs for recycling embeddings
        if rc_inputs is None:
            rc_inputs = {
                'sfea': torch.zeros((1, seq_len, self.c_m), dtype=dtype, device=device),
                'pfea': torch.zeros((1, seq_len, seq_len, self.c_z), dtype=dtype, device=device),
                'cord': torch.zeros((seq_len, N_ATOMS_PER_RESD, 3), dtype=dtype, device=device)
            }
        # calculate the pairwise distance between CB atoms (CA for Glycine)
        cord_mat = get_cb_cords(aa_seq, rc_inputs['cord'])
        dist_mat = cdist(cord_mat.to(torch.float32)).to(dtype)  # cdist() requires FP32 inputs

        # calculate update terms for single features
        sfea_tns_rc = self.norm_m(rc_inputs['sfea'])

        # calculate update terms for pair features
        idxs_mat = torch.clip(torch.floor(
            (dist_mat - self.dist_min) / self.bin_width).to(torch.int64), 0, self.num_bins - 1)
        onht_tns = nn.functional.one_hot(idxs_mat, self.num_bins).unsqueeze(dim=0)

        z = z + self.norm_p(rc_inputs['pfea']) + self.linear(onht_tns.to(dtype))

        # update MSA & pair features
        m = torch.cat(
            [(m[:, 0] + sfea_tns_rc).unsqueeze(dim=1), m[:, 1:]], dim=1)

        return m, z

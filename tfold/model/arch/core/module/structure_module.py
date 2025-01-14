# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
import torch
from torch import nn

from tfold.model.layer import LayerNorm, Linear
from tfold.protein import ProtStruct, ProtConverter
from tfold.transform import rot2quat
from .head import PLDDTHead, TMScoreHead, FrameAngleHead
from .invariant_point_attention import InvariantPointAttention


def init_qta_params(n_smpls, n_resds, num_angles=7, mode='black-hole', dtype=None, device=None):
    """Initialize quaternion-translation-angle (QTA) parameters.

    Args:
        n_smpls: number of samples (default: 1)
        n_resds: number of residues
        mode: initialization mode (choices: 'black-hole' / 'random')
        dtype: (optional) data type
        device: (optional) computational device

    Returns:
        quat_tns: quaternion vectors of size N x L x 4
        trsl_tns: translation vectors of size N x L x 3
        angl_tns: torsion angle matrices of size N x L x K x 2 (K=7)
    """
    if dtype is None:
        dtype = torch.float32

    # initialize quaternion-translation-angle (QTA) parameters
    if mode == 'black-hole':
        quat_tns = torch.cat([
            torch.ones((n_smpls, n_resds, 1), dtype=dtype, device=device),
            torch.zeros((n_smpls, n_resds, 3), dtype=dtype, device=device),
        ], dim=2)
        trsl_tns = torch.zeros((n_smpls, n_resds, 3), dtype=dtype, device=device)
        angl_tns = torch.zeros((n_smpls, n_resds, num_angles, 2), dtype=dtype, device=device)
    elif mode == 'random':
        quat_tns = torch.randn((n_smpls, n_resds, 4), dtype=dtype, device=device)
        quat_tns *= torch.sign(quat_tns[:, :, :1])  # qr: non-negative
        quat_tns /= torch.norm(quat_tns, dim=2, keepdim=True)  # unit L2-norm
        trsl_tns = torch.randn((n_smpls, n_resds, 3), dtype=dtype, device=device)
        angl_tns = torch.zeros((n_smpls, n_resds, num_angles, 2), dtype=dtype, device=device)
    else:
        raise ValueError('unrecognized initialization mode for local frames: {mode}')

    return quat_tns, trsl_tns, angl_tns


class StructureModule(nn.Module):
    """The AlphaFold2 structure module."""

    def __init__(
            self,
            c_s=384,  # number of dimensions in single features
            c_z=256,  # number of dimensions in pair features
            num_layers=8,  # number of layers (all layers share the same set of parameters)
            n_dims_encd=32,  # number of dimensions in positional encodings
            decouple_angle=True,  # whether to predict torsion angles w/ AA-type dependent networks
            tmsc_pred=False,  # whether to predict tmscore
    ):
        super().__init__()
        self.num_layers = num_layers
        self.c_s = c_s
        self.c_z = c_z
        self.n_dims_encd = n_dims_encd
        self.decouple_angle = decouple_angle
        self.tmsc_pred = tmsc_pred

        self.prot_struct = ProtStruct()
        self.prot_converter = ProtConverter()
        self.net = nn.ModuleDict()
        self.net['norm_s'] = LayerNorm(self.c_s)
        self.net['norm_p'] = LayerNorm(self.c_z)
        self.net['linear_s'] = Linear(self.c_s, self.c_s)
        self.net['ipa'] = InvariantPointAttention(c_s=self.c_s, c_z=self.c_z)
        self.net['fa'] = FrameAngleHead(
            c_s=self.c_s,
            n_dims_encd=self.n_dims_encd,
            decouple_angle=self.decouple_angle,
        )
        self.net['plddt'] = PLDDTHead(self.c_s)
        # pTM (and ipTM) predictions
        if self.tmsc_pred:
            self.net['ptm'] = TMScoreHead(self.c_z)

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self,
                aa_seq,
                sfea_tns,
                pfea_tns,
                encd_tns,
                n_lyrs=-1,
                asym_id=None,
                cord_tns=None,
                cmsk_mat=None,
                inter_chain_contact=None,
                ):
        """
        Args:
            aa_seq: amino-acid sequence
            sfea_tns: single features of size N x L x D_s
            pfea_tns: pair features of size N x L x L x D_p
            encd_tns: positional encodings of size N x L x D_e
            n_lyrs: (optional) number of <AF2SMod> layers (-1: default number of layers)
            asym_id: (optional) the asymmetric unit ID (chain ID) of size L (multimer only)
            cord_tns: (optional) initial per-atom 3D coordinates of size L x M x 3
            cmsk_mat: (optional) initial per-atom 3D coordinates' validness masks of size L x M
            inter_chain_contact: (optional) predicted inter-chain contact feature of size N x L x L x 1

        Returns:
        * params_list: list of QTA parameters, one per layer
          > quat: quaternion vectors of size L x 4
          > trsl: translation vectors of size L x 3
          > angl: torsion angle matrices of size L x K x 2
          > quat-u: update signal of quaternion vectors of size L x 4
        * plddt_list: list of per-residue & full-chain lDDT-Ca predictions, one per layer
          > logit: raw classification logits of size L x 50
          > plddt-r: per-residue predicted lDDT-Ca scores of size L
          > plddt-c: full-chain predicted lDDT-Ca score (scalar)
        * cord_list: list of per-atom 3D coordinates of size L x M x 3, one per layer
        * fram_tns_sc: final layer's per-residue side-chain frames of size L x K x 4 x 3
        * tmsc_dict: dict of pTM (and ipTM) predictions
        * sfea_tns: (optional): updated sfea_tns of size N x L x D_s

        Note:
        In <cord_list>, only the last entry contains full-atom 3D coordinates, while all the other
          entries only contain C-Alpha atoms' 3D coordinates.
        """
        n_smpls, n_resds, _ = sfea_tns.shape
        device = sfea_tns.device
        n_lyrs = self.num_layers if n_lyrs == -1 else n_lyrs
        assert n_smpls == 1, f'batch size must be 1 in <AF2SMod>; {n_smpls} detected'

        # pre-process single & pair features
        sfea_tns_init = self.net['norm_s'](sfea_tns)
        pfea_tns = self.net['norm_p'](pfea_tns)
        sfea_tns = self.net['linear_s'](sfea_tns_init)

        # initialize per-residue local frames
        if (cord_tns is None) or (cmsk_mat is None):
            quat_tns, trsl_tns, angl_tns = init_qta_params(n_smpls, n_resds, mode='black-hole', device=device)
        else:
            # build QTA parameters from 3D coordinates
            fram_tns_bb, _, angl_tns, _ = self.prot_converter.cord2fa(aa_seq, cord_tns.float(), cmsk_mat)
            quat_tns = rot2quat(fram_tns_bb[:, 0, :3]).unsqueeze(dim=0)  # N x L x 4
            trsl_tns = fram_tns_bb[:, 0, 3].unsqueeze(dim=0)  # N x L x 3
            # reset certain QTA parameters to the black-hole initialization
            rmsk_vec = ProtStruct.get_atoms(aa_seq, cmsk_mat, ['CA']).to(torch.bool)
            quat_tns_bh, trsl_tns_bh, angl_tns_bh = init_qta_params(n_smpls, n_resds, mode='black-hole', device=device)
            quat_tns = torch.where(rmsk_vec.view(1, -1, 1), quat_tns, quat_tns_bh)
            trsl_tns = torch.where(rmsk_vec.view(1, -1, 1), trsl_tns, trsl_tns_bh)

        params_list, plddt_list, cord_list = [], [], []
        for idx_lyr in range(n_lyrs):
            sfea_tns = self.net['ipa'](sfea_tns, pfea_tns, quat_tns, trsl_tns)
            quat_tns, trsl_tns, angl_tns, quat_tns_upd = \
                self.net['fa'](aa_seq, sfea_tns, sfea_tns_init, encd_tns, quat_tns, trsl_tns)
            plddt_dict = self.net['plddt'](sfea_tns.detach())

            # pack QTA parameters & per-residue lDDT-Ca predictions into dicts
            params = {
                'quat': quat_tns[0],  # L x 4
                'trsl': trsl_tns[0],  # L x 3
                'angl': angl_tns[0],  # L x K x 2
                'quat-u': quat_tns_upd[0],  # L x 4
            }

            # reconstruct per-atom 3D coordinates
            atom_set = 'ca' if idx_lyr != n_lyrs - 1 else 'fa'
            self.prot_struct.init_from_param(aa_seq, params, self.prot_converter, atom_set)

            # record the current layer's predictions
            params_list.append(params)
            plddt_list.append(plddt_dict)
            cord_list.append(self.prot_struct.cord_tns)

            # stop gradient propagation between iterations
            if idx_lyr != n_lyrs - 1:
                quat_tns = quat_tns.detach()

        # obtain side-chain local frames
        self.prot_struct.build_fram_n_angl(self.prot_converter, build_sc=True)
        fram_tns_sc = self.prot_struct.fram_tns_sc

        # obtain pTM (and ipTM) predictions
        tmsc_dict = None
        if self.tmsc_pred:
            tmsc_dict = self.net['ptm'](pfea_tns.detach(), asym_id=asym_id, inter_chain_contact=inter_chain_contact)

        return params_list, plddt_list, cord_list, fram_tns_sc, tmsc_dict

# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
# Data: 2023/11/1 21:52

import torch
import torch.nn as nn
import torch.nn.functional as F

from tfold.model.layer import LayerNorm, Linear
from tfold.model.module.attention import InvariantPointAttention
from tfold.transform.affine import Rigid
from tfold.utils.tensor import dict_multimap
from .protein_mapper import ProteinMapper
from ..head import AngleHead, FrameHead


class ResidualBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.linear_1 = Linear(self.c, self.c, init="relu")
        self.linear_2 = Linear(self.c, self.c, init="relu")
        self.linear_3 = Linear(self.c, self.c, init="final")

        self.relu = nn.ReLU()

    def forward(self, s):
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)

        s = s + s_initial

        return s


class StructureModuleTransition(nn.Module):
    """resnet1d x n + layernorm"""

    def __init__(self,
                 dim,
                 num_layers=1,
                 dropout=0.0,
                 NormLayer=LayerNorm
                 ):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            # residual block
            l = ResidualBlock(self.dim)
            self.layers.append(l)

        self.dropout = nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity()
        self.layer_norm = NormLayer(self.dim)

    def forward(self, s):
        for layer in self.layers:
            s = layer(s)
        s = self.dropout(s)
        s = self.layer_norm(s)

        return s


class StructureModule(ProteinMapper):
    """
    Args:
        c_s: Single representation channel dimension
        c_z: Pair representation channel dimension
        c_ipa_hidden: IPA hidden channel dimension
        c_angle_hidden: Angle resnet (Alg. 23 lines 11-14) hidden channel dimension
        num_heads_ipa: Number of IPA heads
        num_qk_points: Number of query/key points to generate during IPA
        num_v_points:
            Number of value points to generate during IPA
        dropout_rate:
            Dropout rate used throughout the layer
        num_blocks: Number of structure module blocks
        num_transition_blocks: Number of layers in the single representation transition
        num_resnet_blocks:
            Number of blocks in the angle resnet
        num_angles:
            Number of angles to generate in the angle resnet
        trans_scale_factor:
            Scale of single representation transition hidden dimension
    """

    def __init__(
            self,
            c_s: int = 384,
            c_z: int = 256,
            c_ipa: int = 16,
            c_resnet: int = 128,
            no_heads_ipa: int = 12,
            no_qk_points: int = 4,
            no_v_points: int = 8,
            no_blocks: int = 8,
            no_transition_layers: int = 1,
            num_resnet_blocks: int = 2,
            no_angles: int = 7,
            trans_scale_factor: int = 10,
            dropout_rate: float = 0.0,
            eps: float = 1e-8,
            **kwargs
    ):
        super(StructureModule, self).__init__()
        self.eps = eps
        self.c_s = c_s
        self.c_z = c_z
        self.dof = 3
        self.dropout_rate = dropout_rate
        self.num_blocks = no_blocks
        self.trans_scale_factor = trans_scale_factor
        self.c_ipa = c_ipa
        self.num_heads_ipa = no_heads_ipa
        self.num_qk_points = no_qk_points
        self.num_v_points = no_v_points
        self.layer_norm_s = LayerNorm(self.c_s)
        self.layer_norm_z = LayerNorm(self.c_z)

        self.linear_in = Linear(self.c_s, self.c_s)
        self.ipa = InvariantPointAttention(
            self.c_s,
            self.c_z,
            self.c_ipa,
            self.num_heads_ipa,
            self.num_qk_points,
            self.num_v_points
        )
        self.ipa_dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm_ipa = LayerNorm(self.c_s)
        self.no_transition_layers = no_transition_layers
        self.transition = StructureModuleTransition(
            self.c_s,
            self.no_transition_layers,
            self.dropout_rate
        )
        self.bb_update = FrameHead(self.c_s, dof=self.dof)
        self.c_angle_hidden = c_resnet
        self.num_angles = no_angles
        self.num_resnet_blocks = num_resnet_blocks
        self.angle_resnet = AngleHead(
            self.c_s,
            self.c_angle_hidden,
            self.num_resnet_blocks,
            self.num_angles,
            eps=self.eps
        )

    def forward(self, s, aatype, z=None, mask=None):
        """
        Args:
            s: [bs, seq_len, c_s] single representation
            aatype: [bs, seq_len] amino acid indices
            z: [bs, seq_len, seq_len, C_z] pair representation
            mask: Optional [bs, seq_len] sequence mask
        """
        s = self.layer_norm_s(s)
        s_initial = s
        s = self.linear_in(s)
        if z is not None:
            z = self.layer_norm_z(z)

        # [bs, seq_len, 6], quaterion, translation
        rigids = Rigid.identity(
            s.shape[:-1],
            dtype=torch.float32,
            device=s.device,
            fmt="quat"
        )
        outputs = []
        #  fixed z and update s num_block times
        for i in range(self.num_blocks):
            s = s + self.ipa(s, z=z, r=rigids, mask=mask)  # [bs, seq_len, c_s]
            s = self.ipa_dropout(s)
            s = self.layer_norm_ipa(s)
            s = self.transition(s)
            rigids = rigids @ self.bb_update(s)
            backb_to_global = rigids.scale_translation(self.trans_scale_factor)
            # output unnormalized_angles for loss, [bs, seq_len, 7, 2]
            unnormalized_angles = self.angle_resnet(s, s_initial)
            angles = F.normalize(unnormalized_angles, dim=-1, eps=self.eps)
            # convert angle prediction to side chain atom pose
            sidechain_frames = self.torsion_angles_to_frames(
                backb_to_global,
                angles,
                aatype
            )
            pred_xyz = self.frames_and_literature_positions_to_atom14_pos(
                sidechain_frames,
                aatype
            )
            # keep all pose to same class Rigid
            preds = {
                "frames": backb_to_global.to_tensor(self.dof),
                "sidechain_frames": sidechain_frames.to_tensor_4x4(),
                "unnormalized_angles": unnormalized_angles,
                "angles": angles,
                "positions": pred_xyz,
                "states": s
            }
            outputs.append(preds)
            rigids = rigids.stop_rot_gradient()

        outputs = dict_multimap(torch.stack, outputs)  # [bs, ..., num_layers]
        outputs["frames"] = Rigid.from_tensor(outputs["frames"])
        outputs["sidechain_frames"] = Rigid.from_tensor_4x4(outputs["sidechain_frames"])
        outputs["single"] = s
        outputs["pair"] = z

        return outputs

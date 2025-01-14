# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
# Data: 2023/11/13 9:59
import torch
import torch.nn as nn
import torch.nn.functional as F

from tfold.protein import residue_constants as rc
from tfold.transform.affine import Rotation, Rigid


def torsion_angles_to_frames(
        r: Rigid,
        angles: torch.Tensor,
        aatype: torch.Tensor,
        rrgdf: torch.Tensor
):
    """
    make sidechain frames from backbone frames

    Args；
        r: rigid transform, [*, seq_len, dof(7)], quat
        alpha: angles, [*, seq_len, 7, 2]
        aatype: [*, seq_len]
        rrgdf: default frame buffer, [21, 8, 4, 4], 21 is 20 aa types with x

    Returns:
        side chain global pose, [*, seq_len, 8, dof]
    """
    default_4x4 = rrgdf[aatype, ...]  # [bs, seq_len, 8, 4, 4]

    # [*, N, 8] rigid transformations, if rotation representation
    #   One [*, N, 8, 3, 3] rotation matrix and
    #   One [*, N, 8, 3]    translation matrix
    default_r = Rigid.from_tensor_4x4(default_4x4)

    # extend angle to [bs, seq_len, 8, 2]
    bb_rot = angles.new_zeros((*((1,) * len(angles.shape[:-1])), 2))  # [1, 1, 1, 2]
    bb_rot[..., 1] = 1
    bb_rot = bb_rot.expand(*angles.shape[:-2], -1, -1)  # [bs, seq_len, 1, 2]
    angles = torch.cat([bb_rot, angles], dim=-2)  # [bs, seq_len, 8, 2]

    # Produces rotation matrices of the form:
    # [
    #   [1, 0  , 0  ],
    #   [0, a_2,-a_1],
    #   [0, a_1, a_2]
    # ]
    # This follows the original code rather than the supplement, which uses
    # different indices.
    all_rots = angles.new_zeros(default_r.get_rots().shape)  # [bs, seq_len, 8, 3, 3]
    all_rots[..., 0, 0] = 1
    all_rots[..., 1, 1] = angles[..., 1]
    all_rots[..., 1, 2] = -angles[..., 0]
    all_rots[..., 2, 1:] = angles

    all_rots = Rigid(Rotation(rot_mats=all_rots), None)

    all_frames = default_r.compose(all_rots)

    chi2_frame_to_frame = all_frames[..., 5]
    chi3_frame_to_frame = all_frames[..., 6]
    chi4_frame_to_frame = all_frames[..., 7]

    chi1_frame_to_bb = all_frames[..., 4]
    chi2_frame_to_bb = chi1_frame_to_bb.compose(chi2_frame_to_frame)
    chi3_frame_to_bb = chi2_frame_to_bb.compose(chi3_frame_to_frame)
    chi4_frame_to_bb = chi3_frame_to_bb.compose(chi4_frame_to_frame)

    # [bs, seq_len, 8, dof]
    all_frames_to_bb = Rigid.cat(
        [
            all_frames[..., :5],
            chi2_frame_to_bb.unsqueeze(-1),
            chi3_frame_to_bb.unsqueeze(-1),
            chi4_frame_to_bb.unsqueeze(-1),
        ],
        dim=-1
    )
    all_frames_to_global = r[..., None].compose(all_frames_to_bb)

    return all_frames_to_global


def frames_and_literature_positions_to_atom14_pos(
        r: Rigid,
        aatype: torch.Tensor,
        default_frames,
        group_idx,
        atom_mask,
        lit_positions
):
    """
    Args:
        r: side chain rigid transform pose[*, seq_len, 8], 8 is max 8 atom rigid position groups
        aatype: aa seqs [*, seq_len] in range [0, 20] with x
        default_frames； default frame buffer, [21, 8, 4, 4], 21 is 20 aa types with x
        group_idx: residue atom14 groups [21, 14], 14 atom format
        atom_mask: [21, 14], residue 14 atom format mask
        lit_positions: [21, 14, 3], relative atom position

    Return:
        update side chain atoms position, [*, seq_len, 14, 3]
    """
    # [*, N, 14]
    group_mask = group_idx[aatype, ...]

    # [*, N, 14, 8]
    group_mask = F.one_hot(
        group_mask,
        num_classes=default_frames.shape[-3],
    )

    # [bs, seq_len, 14, 8, dof]
    t_atoms_to_global = r[..., None, :] * group_mask

    # [*, N, 14], compose gruop transform
    t_atoms_to_global = t_atoms_to_global.map_tensor_fn(
        lambda x: torch.sum(x, dim=-1)
    )

    # [*, N, 14, 1]
    atom_mask = atom_mask[aatype, ...].unsqueeze(-1)

    # [*, N, 14, 3]
    lit_positions = lit_positions[aatype, ...]
    pred_positions = t_atoms_to_global.apply(lit_positions)
    pred_positions = pred_positions * atom_mask

    return pred_positions


class ProteinMapper(nn.Module):

    def __init__(self):
        super().__init__()

    def _init_residue_constants(self, dtype=torch.float, device=None):
        if not hasattr(self, "default_frames"):
            # 0: backbone-backbone, identity
            # 1: preomega-backbone, identity
            # 2: phi-backbone, (x axis, N, CA)
            # 3: psi-backbone, (C, CA, N)
            # 4-8: chi1-4
            self.register_buffer(
                "default_frames",
                torch.tensor(
                    rc.restype_rigid_group_default_frame,  # [21, 8, 4, 4]
                    dtype=dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False
            )
        if not hasattr(self, "group_idx"):
            self.register_buffer(
                "group_idx",
                torch.tensor(
                    rc.restype_atom14_to_rigid_group,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "atom_mask"):
            self.register_buffer(
                "atom_mask",
                torch.tensor(
                    rc.restype_atom14_mask,
                    dtype=dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "lit_positions"):
            self.register_buffer(
                "lit_positions",
                torch.tensor(
                    rc.restype_atom14_rigid_group_positions,
                    dtype=dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False
            )

    def torsion_angles_to_frames(self, r, alpha, aatype):
        # lazy initialize the residue constants on the correct device
        self._init_residue_constants(alpha.dtype, alpha.device)
        return torsion_angles_to_frames(r, alpha, aatype, self.default_frames)

    def frames_and_literature_positions_to_atom14_pos(
            self, r, aatype  # [*, N, 8]  # [*, N]
    ):
        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(r.get_rots().dtype, r.get_rots().device)
        return frames_and_literature_positions_to_atom14_pos(
            r,
            aatype,
            self.default_frames,
            self.group_idx,
            self.atom_mask,
            self.lit_positions
        )

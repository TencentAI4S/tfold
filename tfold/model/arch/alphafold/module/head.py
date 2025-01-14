# Copyright (c) 2024, Tencent Inc. All rights reserved.
# Data: 2024/5/29 18:08
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from tfold.model.layer import Linear, LayerNorm
from tfold.transform.affine import Rigid, Rotation
from tfold.transform.rotation_conversions import quanternion3_to_4


class ResidualBlock1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.linear_1 = Linear(self.dim, self.dim, init="relu")
        self.linear_2 = Linear(self.dim, self.dim, init="final")
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.relu(x)
        x = self.linear_1(x)

        x = self.relu(x)
        x = self.linear_2(x)

        return x + residual


class AngleHead(nn.Module):
    """
    Implements Algorithm 20, lines 11-14

    Args:
        dim: Input channel dimension
        hidden_dim: Hidden channel dimension
        num_blocks: Number of resnet blocks
        num_angles: Number of torsion angles to generate, default 7,
        eps: normalization eps
    """

    def __init__(self,
                 dim,
                 hidden_dim=128,
                 num_blocks=2,
                 num_angles=7,
                 normalize=False,
                 eps=1e-8,
                 activation=nn.ReLU()):
        super(AngleHead, self).__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.num_angles = num_angles
        self.normalize = normalize
        self.eps = eps
        self.linear_in = Linear(self.dim, self.hidden_dim)
        self.linear_initial = Linear(self.dim, self.hidden_dim)
        self.layers = nn.ModuleList()
        for _ in range(self.num_blocks):
            layer = ResidualBlock1d(self.hidden_dim)
            self.layers.append(layer)

        self.linear_out = Linear(self.hidden_dim, self.num_angles * 2)
        self.act = activation

    def forward(self, s: torch.Tensor, s_initial: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s: [*, dim] single embedding
            s_initial: [*, dim] single embedding as of the start of the StructureModule

        Returns:
            [*, num_angles, 2] predicted angles, in range (0, 1]
        """
        s_initial = self.act(s_initial)
        s_initial = self.linear_initial(s_initial)
        s = self.act(s)
        s = self.linear_in(s)
        s = s + s_initial
        for layer in self.layers:
            s = layer(s)
        s = self.act(s)
        s = self.linear_out(s)  # [*, num_angles * 2]
        s = s.view(s.shape[:-1] + (-1, 2))
        if self.normalize:
            s = F.normalize(s, dim=-1, eps=self.eps)

        return s


class FrameHead(nn.Module):
    """
    Implements part of Algorithm 23.

    Args:
        decouple: whether decouple rotation and translation linear
    """

    def __init__(self, c_s, dof=3, decouple=False):
        """
        Args:
            c_s: Single representation channel dimension
        """
        super(FrameHead, self).__init__()
        assert dof in (3, 4, 6), f"only support quaternion vector size 3 or 4, or ortho6d vector"
        self.c_s = c_s
        self.decouple = decouple
        self.dof = dof
        if decouple:
            self.rotation_linear = Linear(self.c_s, dof, init="final")
            self.translation_linear = Linear(self.c_s, 3, init="final")
        else:
            self.linear = Linear(self.c_s, dof + 3, init="final")

    def forward(self, s: torch.Tensor) -> Rigid:
        """
        Args:
            s； [*, c_s] single representation

        Returns:
            [*, dof + 3] update vector
        """
        if self.decouple:
            rotations = self.rotation_linear(s)
            translations = self.translation_linear(s)
        else:
            rot = self.linear(s)
            rotations = rot[..., :self.dof]
            translations = rot[..., self.dof:]

        # auto convert rigid to float32
        if self.dof in (3, 4):
            quanterions = rotations
            if self.dof == 3:
                quanterions = quanternion3_to_4(quanterions)
            return Rigid(Rotation(rot_mats=None, quats=quanterions), translations)
        else:
            return Rigid.from_tensor_9(ortho6d=rotations, trans=translations)


class DistogramHead(nn.Module):
    """
    Computes a distogram probability distribution.

    For use in computation of distogram loss, subsection 1.9.8
    """

    def __init__(self, c_z, num_bins=64, normalize=False):
        """
        Args:
            c_z:
                Input channel dimension
            num_bins:
                Number of distogram bins
        """
        super().__init__()
        self.c_z = c_z
        self.num_bins = num_bins
        self.normalize = normalize
        self.linear = Linear(self.c_z, self.num_bins, init="final")

    def forward(self, z):
        """
        Args:
            z: [*, seq_len, seq_len, C_z] pair embedding
        Returns:
            [*, seq_len, seq_len, num_bins] distogram probability distribution
        """
        logits = self.linear(z)
        logits = logits + logits.transpose(-2, -3)
        if self.normalize:
            logits = logits / 2

        return logits

    @staticmethod
    def compute_distogram(positions, num_bins=15, min_bin=3.375, max_bin=21.375):
        """
        Args；
            coords： [bs, seq_len, 3 atoms, 3], where it's [N, CA, C] x 3 coordinates.

        Returns:
            dist bins: [bs, seq_len, seq_len], range [0, 14]
        """
        assert positions.shape[-2] == 3, f"only support bacbone atoms"
        boundaries = torch.linspace(
            min_bin,
            max_bin,
            num_bins - 1,
            device=positions.device
        )
        boundaries = boundaries ** 2
        N, CA, C = [x.squeeze(-2) for x in positions.chunk(3, dim=-2)]
        # Infer CB coordinates.
        b = CA - N
        c = C - CA
        a = b.cross(c, dim=-1)
        CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA
        dists = (CB[..., None, :, :] - CB[..., :, None, :]).pow(2).sum(dim=-1, keepdims=True)

        bins = torch.sum(dists > boundaries, dim=-1)

        return bins


class ExperimentallyResolvedHead(nn.Module):
    """
    For use in computation of "experimentally resolved" loss, subsection
    1.9.10
    Args:
        c_s: Input channel dimension
        c_out: Number of distogram bins
    """

    def __init__(self, c_s, c_out=37):
        super(ExperimentallyResolvedHead, self).__init__()
        self.c_s = c_s
        self.c_out = c_out
        self.linear = Linear(self.c_s, self.c_out, init="final")

    def forward(self, s):
        """
        Args:
            s: [*, N_res, C_s] single embedding
        Returns: [*, N, C_out] logits
        """
        logits = self.linear(s)

        return logits


class LDDTHead(nn.Module):
    """per residue lDDT-Ca scores prediction"""

    def __init__(self,
                 c_in=384,
                 c_hidden=None,
                 num_bins=50,
                 num_atoms=1
                 ):
        super().__init__()
        self.c_in = c_in
        self.num_bins = num_bins
        self.num_atoms = num_atoms
        self.hidden_dim = c_hidden or c_in
        self.relu = nn.ReLU()
        self.layer_norm = LayerNorm(self.c_in)
        self.linear_1 = Linear(self.c_in, self.hidden_dim, init="relu")
        self.linear_2 = Linear(self.hidden_dim, self.hidden_dim, init="relu")
        self.linear_3 = Linear(self.hidden_dim, self.num_bins * num_atoms, init="final")

    @classmethod
    def compute_plddt(cls, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [*, num_bins]

        Returns:
            plddt: [*]
        """
        num_bins = logits.shape[-1]
        bin_width = 1.0 / num_bins
        bounds = torch.arange(0.5 * bin_width, 1.0, step=bin_width, dtype=logits.dtype,
                              device=logits.device)  # [num_bins, ]
        pred_lddt_ca = (logits.softmax(dim=-1) @ bounds[:, None]).squeeze(-1)

        return pred_lddt_ca * 100

    def forward(self, x):
        """
        Args:
            x: [*, dim], single features

        Returns:
            [*, num_atoms, num_bins], predict per-residue & full-chain lDDT-Ca scores
        """
        x = self.layer_norm(x)
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        x = self.relu(x)
        logits = self.linear_3(x)
        return logits.reshape(logits.shape[:-1] + (self.num_atoms, self.num_bins))


class MaskedMSAHead(nn.Module):
    """
    For use in computation of masked MSA loss, subsection 1.9.9
    Args:
        c_m: MSA channel dimension
        num_classes: Output channel dimension
    """

    def __init__(self, c_m, num_classes=23):
        super(MaskedMSAHead, self).__init__()
        self.c_m = c_m
        self.num_classes = num_classes
        self.linear = Linear(self.c_m, self.num_classes, init="final")

    def forward(self, m):
        """
        Args:
            m: [*, N_seq, N_res, C_m] MSA embedding

        Returns:
            [*, N_seq, N_res, C_out] reconstruction
        """
        logits = self.linear(m)
        return logits


def _calculate_bin_centers(boundaries: torch.Tensor):
    step = boundaries[1] - boundaries[0]
    bin_centers = boundaries + step / 2
    bin_centers = torch.cat(
        [bin_centers, (bin_centers[-1] + step).unsqueeze(-1)], dim=0
    )
    return bin_centers


def _calculate_expected_aligned_error(
        alignment_confidence_breaks: torch.Tensor,
        aligned_distance_error_probs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bin_centers = _calculate_bin_centers(alignment_confidence_breaks)
    return (
        torch.sum(aligned_distance_error_probs * bin_centers, dim=-1),
        bin_centers[-1],
    )


class TMScoreHead(nn.Module):
    """
    For use in computation of TM-score, subsection 1.9.7

    Args:
        c_z: Input channel dimension
        num_bins: Number of bins
    """

    def __init__(self, c_z, num_bins=64):
        super().__init__()
        self.c_z = c_z
        self.num_bins = num_bins
        self.linear = Linear(self.c_z, self.num_bins, init="final")

    def forward(self, z):
        """
        Args:
            z: [*, seq_len, seq_len, c_z] pairwise embedding

        Returns:
            [*, seq_len, seq_len, no_bins] prediction
        """
        # [*, N, N, num_bins]
        logits = self.linear(z)
        return logits

    @classmethod
    def compute_tm_score(
            cls,
            logits: torch.Tensor,
            asym_id: Optional[torch.Tensor] = None,
            residue_weights: Optional[torch.Tensor] = None,
            max_bin: int = 31,
            num_bins: int = 64,
            eps: float = 1e-8
    ) -> torch.Tensor:
        """Computes pTM and ipTM from logits.

        Args；
            logits: [*, seq_len, seq_len, num_bins], pairwise prediction
            residue_weights: [*, seq_len] the per-residue weights to use for the expectation
            asym_id: [*, seq_len] the asymmetric unit ID - the chain ID. Only needed for ipTM calculation

        Returns:
            score: the predicted TM alignment or the predicted iTM score.
        """
        if residue_weights is None:
            residue_weights = logits.new_ones(logits.shape[-2])

        boundaries = torch.linspace(
            0, max_bin, steps=(num_bins - 1), device=logits.device
        )

        bin_centers = _calculate_bin_centers(boundaries)
        clipped_n = max(torch.sum(residue_weights), 19)

        d0 = 1.24 * (clipped_n - 15) ** (1.0 / 3) - 1.8

        probs = logits.softmax(dim=-1)

        tm_per_bin = 1.0 / (1 + (bin_centers ** 2) / (d0 ** 2))
        predicted_tm_term = torch.sum(probs * tm_per_bin, dim=-1)

        if asym_id is not None:
            seq_len = residue_weights.shape[-1]
            pair_mask = residue_weights.new_ones((seq_len, seq_len), dtype=torch.int32)
            if len(asym_id.shape) > 1:
                assert len(asym_id.shape) <= 2
                batch_size = asym_id.shape[0]
                pair_mask = residue_weights.new_ones((batch_size, seq_len, seq_len), dtype=torch.int32)
            pair_mask *= (asym_id[..., None] != asym_id[..., None, :]).to(dtype=pair_mask.dtype)
            predicted_tm_term *= pair_mask
            pair_residue_weights = pair_mask * (
                    residue_weights[..., None, :] * residue_weights[..., :, None]
            )
            denom = eps + torch.sum(pair_residue_weights, dim=-1, keepdims=True)
            normed_residue_mask = pair_residue_weights / denom
        else:
            normed_residue_mask = residue_weights / (eps + residue_weights.sum())

        per_alignment = torch.sum(predicted_tm_term * normed_residue_mask, dim=-1)
        weighted = per_alignment * residue_weights
        argmax = (weighted == torch.max(weighted)).nonzero()[0]
        return per_alignment[tuple(argmax)]

    @classmethod
    def compute_weighted_ptm_score(cls,
                                   logits: torch.Tensor,
                                   asym_id: Optional[torch.Tensor] = None,
                                   residue_weights: Optional[torch.Tensor] = None,
                                   max_bin: int = 31,
                                   num_bins: int = 64,
                                   iptm_weight: float = 0.8,
                                   ptm_weight: float = 0.2,
                                   eps: float = 1e-8
                                   ):
        scores = {}
        scores["ptm"] = cls.compute_tm_score(
            logits, residue_weights=residue_weights, max_bin=max_bin, num_bins=num_bins, eps=eps
        )

        if asym_id is not None:
            scores["iptm"] = cls.compute_tm_score(logits, asym_id=asym_id, residue_weights=residue_weights,
                                                  max_bin=max_bin, num_bins=num_bins, eps=eps)
            scores["weighted_ptm"] = iptm_weight * scores["ptm"] + ptm_weight * scores["iptm"]
        return scores

    @classmethod
    def compute_predicted_aligned_error(
            cls,
            logits: torch.Tensor,
            max_bin: int = 31,
            num_bins: int = 64
    ) -> Dict[str, torch.Tensor]:
        """Computes aligned confidence metrics from logits.

        Args:
              logits: [*, num_res, num_res, num_bins] the logits output from
                PredictedAlignedErrorHead.
              max_bin: Maximum bin value
              num_bins: Number of bins
        Returns:
              aligned_confidence_probs: [*, num_res, num_res, num_bins] the predicted
                aligned error probabilities over bins for each residue pair.
              predicted_aligned_error: [*, num_res, num_res] the expected aligned distance
                error for each pair of residues.
              max_predicted_aligned_error: [*] the maximum predicted error possible.
        """
        boundaries = torch.linspace(
            0, max_bin, steps=(num_bins - 1), device=logits.device
        )
        aligned_confidence_probs = torch.nn.functional.softmax(logits, dim=-1)

        (
            predicted_aligned_error,
            max_predicted_aligned_error,
        ) = _calculate_expected_aligned_error(
            alignment_confidence_breaks=boundaries,
            aligned_distance_error_probs=aligned_confidence_probs,
        )

        return {
            "aligned_confidence_probs": aligned_confidence_probs,
            "predicted_aligned_error": predicted_aligned_error,
            "max_predicted_aligned_error": max_predicted_aligned_error
        }

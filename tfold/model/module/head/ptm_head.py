"""The network for predicting TM-score."""
import torch
from torch import nn


def calc_bin_centers(boundaries: torch.Tensor):
    """Gets the bin centers from the bin edges.

    Args:
        boundaries: [num_bins - 1] the error bin edges.

    Returns:
        bin_centers: [num_bins] the error bin centers.
    """
    step = boundaries[1] - boundaries[0]
    # Add half-step to get the center
    bin_centers = boundaries + step / 2
    # Add a catch-all bin at the end.
    bin_centers = torch.cat(
        [bin_centers, (bin_centers[-1] + step).unsqueeze(-1)], dim=0
    )
    return bin_centers


def compute_ptmscore(logits, max_bin=31, no_bins=64, residue_weights=None, asym_id=None,
                     interface=False, pairwise=False):
    """Compute predicted TMscore or predicted interface TMscore
    Args:
        logits: the logits output from ptm_net: [L, L, n_bins]
        max_bin: the max distance of error bins
        no_bins: the number of error bins
        residue_weights: the per residue weight to use for the exceptation: [L]
        asym_id: the asymmetric unit ID - the chain ID. Only needed for ipTM calculation (when interface=True)
        interface: If true, interface predicted TMscore is computed
        pairwise: If true, only two chain will be considered, the last chain and the others (consider as one chain)

    Returns:
        ptm_score: The predicted TMscore (ptm) or the predicted interface TMscore (iptm)
    """
    L = logits.shape[-2]
    if residue_weights is None:
        residue_weights = torch.ones((L), device=logits.device)

    boundaries = torch.linspace(0, max_bin, steps=(no_bins - 1), device=logits.device)
    bin_centers = calc_bin_centers(boundaries)

    num_res = torch.sum(residue_weights)
    clipped_n = max(num_res, 19)

    # Computed d_0 as defined by TMscore
    d0 = 1.24 * (clipped_n - 15) ** (1.0 / 3) - 1.8

    # convert logits to probs
    probs = torch.nn.functional.softmax(logits, dim=-1)

    # Tmscore term for every bin
    tm_per_bin = 1.0 / (1 + (bin_centers ** 2) / (d0 ** 2))
    predicted_tm_term = torch.sum(probs * tm_per_bin, dim=-1)
    pair_mask = torch.ones((L, L), device=logits.device)

    # for multimer interface predicted TMscore
    if interface and asym_id is not None:
        if pairwise:
            last_chn_id = torch.max(asym_id)
            pair_mask = (asym_id != last_chn_id)[:, None] == (asym_id == last_chn_id)[None, :]
        else:
            pair_mask *= asym_id[:, None] != asym_id[None, :]

    predicted_tm_term *= pair_mask
    pair_residue_weights = pair_mask * (residue_weights[None, :] * residue_weights[:, None])

    normed_residue_mask = pair_residue_weights / (1e-8 + torch.sum(pair_residue_weights, dim=-1, keepdim=True))

    per_alignment = torch.sum(predicted_tm_term * normed_residue_mask, dim=-1)

    weighted = per_alignment * residue_weights
    argmax = (weighted == torch.max(weighted)).nonzero()[0]

    return per_alignment[tuple(argmax)]


class TMScoreHead(nn.Module):
    """The network for predicting TM-score."""

    def __init__(self, c_z=256, num_bins=64):
        super().__init__()
        self.c_z = c_z
        self.num_bins = num_bins
        self.net = nn.ModuleDict()
        self.net['tm'] = nn.Sequential(
            nn.Linear(self.c_z, self.c_z),
            nn.ReLU(),
            nn.Linear(self.c_z, self.c_z),
            nn.ReLU(),
            nn.Linear(self.c_z, self.num_bins),
        )

    def forward(self, z, asym_id=None):
        """
        Args:
            z: [N, L, L, c_z], pair features
            asym_id: (optional) the asymmetric unit ID (chain ID) of size L (multimer only)

        Returns:
            tmsc_dict: dict of pTM (and ipTM) predictions
        """

        # calculate the pTM score
        logits = self.net['tm'](z)  # N x L x L x C
        ptm = compute_ptmscore(logits[0])
        tmsc_dict = {
            'ptm_logt': logits[0],
            'ptm': ptm,
        }

        # calculate the ipTM score (and overall confidence)
        if asym_id is not None:
            asym_id = asym_id.to(z.device)
            iptm = compute_ptmscore(logits[0], asym_id=asym_id, interface=True)
            tmsc_dict['iptm'] = iptm
            tmsc_dict['ranking_confidence'] = 0.8 * iptm + 0.2 * ptm

        return tmsc_dict

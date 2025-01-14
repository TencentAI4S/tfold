# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
import random

import torch

from .prot_constants import RESD_NAMES_1C


def get_mlm_masks(aa_seqs, mask_prob=0.15, mask_vecs=None):
    """Get random masks for masked language modeling.

    Notes:
        For identical amino-acid sequences, their random masks will be the same, thus ensuring no
        label leakage for masked language modeling.

    Args:
        aa_seqs: list of amino-acid sequences, each of length L_i
        mask_prob: (optional) how likely one token is masked out
        mask_vecs: (optional) list of masking-allowed-or-not indicators, each of length L_i

    Returns:
        mask_vec: random masks of size L (L = \sum_i L_i)
    """
    # generate random masks for each unique amino-acid sequence
    mask_vec_dict = {}
    for seq_idx, aa_seq in enumerate(aa_seqs):
        if aa_seq in mask_vec_dict:  # do not re-generate random masks for the same sequence
            continue

        if mask_vecs is None:
            idxs_resd_cand = list(range(len(aa_seq)))  # all the residues are candidates
        else:
            idxs_resd_cand = torch.nonzero(mask_vecs[seq_idx])[:, 0].tolist()

        n_resds_cand = len(idxs_resd_cand)  # number of candidate residues
        n_resds_mask = int(n_resds_cand * mask_prob + 0.5)  # number of masked residues
        idxs_resd = random.sample(idxs_resd_cand, n_resds_mask)  # indices of masked residues
        mask_vec = torch.zeros(len(aa_seq), dtype=torch.int8)
        mask_vec[idxs_resd] = 1  # 1: masked-out token
        mask_vec_dict[aa_seq] = mask_vec

    # concatenate random masks into one
    mask_vec = torch.cat([mask_vec_dict[x] for x in aa_seqs], dim=0)

    return mask_vec


def apply_mlm_masks(tokn_mat_orig, mask_mat, alphabet):
    """Apply random masks for masked language modeling.

    Notes:
        For each token to be masked, it has an 80% probability of being replaced with a mask token,
        a 10% probability of being replaced with a random token (20 standard AA types), and an 10%
        probability of being replaced with an unmasked token.

    Args:
        tokn_mat_orig: original token indices of size N x L
        mask_mat: random masks of size N x L
        alphabet: alphabet used for tokenization

    Returns:
        tokn_mat_pert: perturbed token indices of size N x L
    """
    dtype = tokn_mat_orig.dtype
    device = tokn_mat_orig.device
    batch_size, seq_len = tokn_mat_orig.shape

    # generate perturbed token indices
    toks = random.choices(RESD_NAMES_1C, k=(batch_size * seq_len))
    prob_mat = torch.rand((batch_size, seq_len), dtype=torch.float32, device=device)
    mask_mat_pri = (mask_mat * torch.lt(prob_mat, 0.8)).to(torch.bool)
    mask_mat_sec = (mask_mat * torch.gt(prob_mat, 0.9)).to(torch.bool)
    tokn_mat_pri = alphabet.mask_idx * torch.ones_like(tokn_mat_orig)
    tokn_mat_sec = torch.tensor(
        [alphabet.get_idx(x) for x in toks], dtype=dtype, device=device).view(batch_size, seq_len)
    tokn_mat_pert = torch.where(
        mask_mat_pri, tokn_mat_pri, torch.where(mask_mat_sec, tokn_mat_sec, tokn_mat_orig))

    return tokn_mat_pert

# -*- coding: utf-8 -*-
# Copyright (c) 2023, Tencent Inc. All rights reserved.
import random
import re

import torch

from tfold.utils.tensor import cdist
from .prot_constants import RESD_NAMES_1C
from .prot_struct import ProtStruct


def get_chain_ids(inputs):
    """Get a list of chain IDs.

    Args:
        inputs: dict of input tensors (as returned by various datasets)

    Returns:
        chain_ids: list of chain IDs (sorted in the alphabetical ordering)

    Notes:
        It is assumed that each chain ID consists of one or more alphanumeric characters.
    """

    regex = re.compile(r'^(base|[0-9A-Za-z]+(-[0-9A-Za-z]+)+)$')
    chain_ids = [x for x in inputs if not re.search(regex, x)]

    return chain_ids


def get_complex_id(inputs):
    """Get ligand ID and receptor ID.

    Args:
        inputs: dict of input tensors (as returned by various datasets)

    Returns:
        ligand_id: chain ID of ligand
        receptor_id: chain ID of receptor, None is not exists
    """

    complex_id = [key for key in inputs.keys() if ':' in key]
    assert len(complex_id) in (0, 1), f'unexpected number of chains in complex_id: {" ".join(complex_id)}'
    if len(complex_id) == 0:
        receptor_id = None
        regex = re.compile(r'^(base|[0-9A-Za-z]+(-[0-9A-Za-z]+)+)$')
        chain_ids = [x for x in inputs if not re.search(regex, x)]
        ligand_id = '-'.join(chain_ids)
    else:
        ligand_id, receptor_id = complex_id[0].split(':')

    assert ligand_id in inputs, f'ligand_id: {ligand_id} should be in inputs'

    return ligand_id, receptor_id


def get_asym_ids(aa_seqs):
    """Get asymmetric IDs.

    Args:
        aa_seqs: list of amino-acid sequences, each of length L_i

    Returns:
        asym_ids: asymmetric IDs of size L (L = \sum_i L_i)
    """

    seq_lens = torch.tensor([len(x) for x in aa_seqs], dtype=torch.int32)
    asym_ids = torch.arange(len(aa_seqs)).repeat_interleave(seq_lens) + 1

    return asym_ids


def get_enty_ids(aa_seqs):
    """Get entity IDs.

    Args:
        aa_seqs: list of amino-acid sequences, each of length L_i

    Returns:
        enty_ids: entity IDs of size L (L = \sum_i L_i)
    """
    aa_seqs_uniq = sorted(list(set(aa_seqs)))
    seq_lens = torch.tensor([len(x) for x in aa_seqs], dtype=torch.int32)
    enty_ids = torch.tensor(
        [aa_seqs_uniq.index(x) for x in aa_seqs]).repeat_interleave(seq_lens) + 1

    return enty_ids


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
    for idx_chn, aa_seq in enumerate(aa_seqs):
        if aa_seq in mask_vec_dict:  # do not re-generate random masks for the same sequence
            continue
        if mask_vecs is None:
            idxs_resd_cand = list(range(len(aa_seq)))  # all the residues are candidates
        else:
            idxs_resd_cand = torch.nonzero(mask_vecs[idx_chn])[:, 0].tolist()
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


def calc_ppi_sites(prot_data, chn_ids, dist_thres=10):
    """Calculate PPI sites"""

    assert len(chn_ids) == 2

    chn_data_pri = prot_data[chn_ids[0]]
    chn_data_sec = prot_data[chn_ids[1]]
    cord_mat_pri = ProtStruct.get_atoms(chn_data_pri['seq'], chn_data_pri['cord'], ['CA'])
    cmsk_vec_pri = ProtStruct.get_atoms(chn_data_pri['seq'], chn_data_pri['cmsk'], ['CA'])
    cord_mat_sec = ProtStruct.get_atoms(chn_data_sec['seq'], chn_data_sec['cord'], ['CA'])
    cmsk_vec_sec = ProtStruct.get_atoms(chn_data_sec['seq'], chn_data_sec['cmsk'], ['CA'])
    dist_mat = cdist(cord_mat_pri, cord_mat_sec)
    dist_max = torch.max(dist_mat)
    dist_mat += dist_max * (1 - torch.outer(cmsk_vec_pri, cmsk_vec_sec))

    dist_vec_pri = torch.min(dist_mat, dim=1)[0]  # minimal distance to the secondary chain
    dist_vec_sec = torch.min(dist_mat, dim=0)[0]  # minimal distance to the primary chain
    ppi_data = torch.cat(
        [torch.lt(dist_vec_pri, dist_thres).to(torch.int8), torch.lt(dist_vec_sec, dist_thres).to(torch.int8)], dim=0)

    return ppi_data


def calc_inter_contacts(prot_data, chn_ids, dist_thres=10):
    """Calculate contacts"""

    assert len(chn_ids) == 2
    chn_data_pri = prot_data[chn_ids[0]]
    chn_data_sec = prot_data[chn_ids[1]]
    len_pri = len(chn_data_pri['seq'])
    cord_mat_pri = ProtStruct.get_atoms(chn_data_pri['seq'], chn_data_pri['cord'], ['CA'])
    cmsk_vec_pri = ProtStruct.get_atoms(chn_data_pri['seq'], chn_data_pri['cmsk'], ['CA'])
    cord_mat_sec = ProtStruct.get_atoms(chn_data_sec['seq'], chn_data_sec['cord'], ['CA'])
    cmsk_vec_sec = ProtStruct.get_atoms(chn_data_sec['seq'], chn_data_sec['cmsk'], ['CA'])

    cord_mat = torch.cat([cord_mat_pri, cord_mat_sec], dim=0)
    cmsk_vec = torch.cat([cmsk_vec_pri, cmsk_vec_sec], dim=0)
    dist_mat = cdist(cord_mat, cord_mat)
    dist_max = torch.max(dist_mat)
    dist_mat += dist_max * (1 - torch.outer(cmsk_vec, cmsk_vec))

    contacts_data = torch.lt(dist_mat, dist_thres).to(torch.int8)
    contacts_data[:len_pri, :len_pri].zero_()
    contacts_data[len_pri:, len_pri:].zero_()

    return contacts_data

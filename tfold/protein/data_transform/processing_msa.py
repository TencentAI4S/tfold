# Copyright (c) 2024, Tencent Inc. All rights reserved.
# Data: 2024/5/29 16:59
from functools import reduce
from operator import add

import numpy as np
import torch

from .utils import curry1, make_one_hot, shaped_categorical
from .. import residue_constants as rc

MSA_FEATURE_NAMES = [
    "msa",
    "deletion_matrix",
    "msa_mask",
    "msa_row_mask",
    "bert_mask",
    "true_msa",
]


@curry1
def sample_msa(protein, max_seq, keep_extra, seed=None):
    """Sample MSA randomly, remaining sequences are stored are stored as `extra_*`."""
    num_seq = protein["msa"].shape[0]

    g = None
    if seed is not None:
        g = torch.Generator(device=protein["msa"].device)
        g.manual_seed(seed)

    shuffled = torch.randperm(num_seq - 1, generator=g) + 1
    index_order = torch.cat(
        (torch.tensor([0], device=shuffled.device), shuffled),
        dim=0
    )
    num_sel = min(max_seq, num_seq)
    sel_seq, not_sel_seq = torch.split(
        index_order, [num_sel, num_seq - num_sel]
    )

    for k in MSA_FEATURE_NAMES:
        if k in protein:
            if keep_extra:
                protein["extra_" + k] = torch.index_select(
                    protein[k], 0, not_sel_seq
                )
            protein[k] = torch.index_select(protein[k], 0, sel_seq)

    return protein


def make_msa_mask(protein):
    """Mask features are all ones, but will later be zero-padded."""
    protein["msa_mask"] = torch.ones(protein["msa"].shape, dtype=torch.float32)
    protein["msa_row_mask"] = torch.ones(
        (protein["msa"].shape[0]), dtype=torch.float32
    )
    return protein


@curry1
def make_masked_msa(protein, config, replace_fraction):
    """Create data for BERT on raw MSA."""
    # Add a random amino acid uniformly.
    random_aa = torch.tensor(
        [0.05] * 20 + [0.0, 0.0],
        dtype=torch.float32,
        device=protein["aatype"].device
    )

    categorical_probs = (
            config.uniform_prob * random_aa
            + config.profile_prob * protein["hhblits_profile"]
            + config.same_prob * make_one_hot(protein["msa"], 22)
    )

    # Put all remaining probability on [MASK] which is a new column
    pad_shapes = list(
        reduce(add, [(0, 0) for _ in range(len(categorical_probs.shape))])
    )
    pad_shapes[1] = 1
    mask_prob = (
            1.0 - config.profile_prob - config.same_prob - config.uniform_prob
    )
    assert mask_prob >= 0.0

    categorical_probs = torch.nn.functional.pad(
        categorical_probs, pad_shapes, value=mask_prob
    )

    sh = protein["msa"].shape
    mask_position = torch.rand(sh) < replace_fraction

    bert_msa = shaped_categorical(categorical_probs)
    bert_msa = torch.where(mask_position, bert_msa, protein["msa"])

    # Mix real and masked MSA
    protein["bert_mask"] = mask_position.to(torch.float32)
    protein["true_msa"] = protein["msa"]
    protein["msa"] = bert_msa

    return protein


@curry1
def nearest_neighbor_clusters(protein, gap_agreement_weight=0.0):
    weights = torch.cat(
        [
            torch.ones(21, device=protein["msa"].device),
            gap_agreement_weight * torch.ones(1, device=protein["msa"].device),
            torch.zeros(1, device=protein["msa"].device)
        ],
        0,
    )

    # Make agreement score as weighted Hamming distance
    msa_one_hot = make_one_hot(protein["msa"], 23)
    sample_one_hot = protein["msa_mask"][:, :, None] * msa_one_hot
    extra_msa_one_hot = make_one_hot(protein["extra_msa"], 23)
    extra_one_hot = protein["extra_msa_mask"][:, :, None] * extra_msa_one_hot

    num_seq, num_res, _ = sample_one_hot.shape
    extra_num_seq, _, _ = extra_one_hot.shape

    # Compute tf.einsum('mrc,nrc,c->mn', sample_one_hot, extra_one_hot, weights)
    # in an optimized fashion to avoid possible memory or computation blowup.
    agreement = torch.matmul(
        torch.reshape(extra_one_hot, [extra_num_seq, num_res * 23]),
        torch.reshape(
            sample_one_hot * weights, [num_seq, num_res * 23]
        ).transpose(0, 1),
    )

    # Assign each sequence in the extra sequences to the closest MSA sample
    protein["extra_cluster_assignment"] = torch.argmax(agreement, dim=1).to(
        torch.int64
    )

    return protein


def unsorted_segment_sum(data, segment_ids, num_segments):
    """
    Computes the sum along segments of a tensor. Similar to
    tf.unsorted_segment_sum, but only supports 1-D indices.

    Args:
        data: A tensor whose segments are to be summed.
        segment_ids: The 1-D segment indices tensor.
        num_segments: The number of segments.

    Returns:
        A tensor of same data type as the data argument.
    """
    assert (
            len(segment_ids.shape) == 1 and
            segment_ids.shape[0] == data.shape[0]
    )
    segment_ids = segment_ids.view(
        segment_ids.shape[0], *((1,) * len(data.shape[1:]))
    )
    segment_ids = segment_ids.expand(data.shape)
    shape = [num_segments] + list(data.shape[1:])
    tensor = (
        torch.zeros(*shape, device=segment_ids.device)
        .scatter_add_(0, segment_ids, data.float())
    )
    tensor = tensor.type(data.dtype)
    return tensor


@curry1
def summarize_clusters(protein):
    """Produce profile and deletion_matrix_mean within each cluster."""
    num_seq = protein["msa"].shape[0]

    def csum(x):
        return unsorted_segment_sum(
            x, protein["extra_cluster_assignment"], num_seq
        )

    mask = protein["extra_msa_mask"]
    mask_counts = 1e-6 + protein["msa_mask"] + csum(mask)  # Include center

    msa_sum = csum(mask[:, :, None] * make_one_hot(protein["extra_msa"], 23))
    msa_sum += make_one_hot(protein["msa"], 23)  # Original sequence
    protein["cluster_profile"] = msa_sum / mask_counts[:, :, None]
    del msa_sum

    del_sum = csum(mask * protein["extra_deletion_matrix"])
    del_sum += protein["deletion_matrix"]  # Original sequence
    protein["cluster_deletion_mean"] = del_sum / mask_counts
    del del_sum

    return protein


@curry1
def crop_extra_msa(protein, max_extra_msa):
    num_seq = protein["extra_msa"].shape[0]
    num_sel = min(max_extra_msa, num_seq)
    select_indices = torch.randperm(num_seq)[:num_sel]
    for k in MSA_FEATURE_NAMES:
        if "extra_" + k in protein:
            protein["extra_" + k] = torch.index_select(
                protein["extra_" + k], 0, select_indices
            )

    return protein


def delete_extra_msa(protein):
    for k in MSA_FEATURE_NAMES:
        if "extra_" + k in protein:
            del protein["extra_" + k]
    return protein


def correct_msa_restypes(protein):
    """Correct MSA restype to have the same order as rc."""
    new_order_list = rc.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE
    new_order = torch.tensor(
        [new_order_list] * protein["msa"].shape[1],
        device=protein["msa"].device,
    ).transpose(0, 1)
    protein["msa"] = torch.gather(new_order, 0, protein["msa"])

    perm_matrix = np.zeros((22, 22), dtype=np.float32)
    perm_matrix[range(len(new_order_list)), new_order_list] = 1.0

    for k in protein:
        if "profile" in k:
            num_dim = protein[k].shape.as_list()[-1]
            assert num_dim in [
                20,
                21,
                22,
            ], "num_dim for %s out of expected range: %s" % (k, num_dim)
            protein[k] = torch.dot(protein[k], perm_matrix[:num_dim, :num_dim])

    return protein


def make_hhblits_profile(protein):
    """Compute the HHblits MSA profile if not already present."""
    if "hhblits_profile" in protein:
        return protein

    # Compute the profile for every residue (over all MSA sequences).
    msa_one_hot = make_one_hot(protein["msa"], 22)

    protein["hhblits_profile"] = torch.mean(msa_one_hot, dim=0)
    return protein


@curry1
def make_msa_feat(protein):
    """Create and concatenate MSA features."""
    # Whether there is a domain break. Always zero for chains, but keeping for
    # compatibility with domain datasets.
    has_break = torch.clip(
        protein["between_segment_residues"].to(torch.float32), 0, 1
    )
    aatype_1hot = make_one_hot(protein["aatype"], 21)

    target_feat = [
        torch.unsqueeze(has_break, dim=-1),
        aatype_1hot,  # Everyone gets the original sequence.
    ]

    msa_1hot = make_one_hot(protein["msa"], 23)
    has_deletion = torch.clip(protein["deletion_matrix"], 0.0, 1.0)
    deletion_value = torch.atan(protein["deletion_matrix"] / 3.0) * (
            2.0 / np.pi
    )

    msa_feat = [
        msa_1hot,
        torch.unsqueeze(has_deletion, dim=-1),
        torch.unsqueeze(deletion_value, dim=-1),
    ]

    if "cluster_profile" in protein:
        deletion_mean_value = torch.atan(
            protein["cluster_deletion_mean"] / 3.0
        ) * (2.0 / np.pi)
        msa_feat.extend(
            [
                protein["cluster_profile"],
                torch.unsqueeze(deletion_mean_value, dim=-1),
            ]
        )

    if "extra_deletion_matrix" in protein:
        protein["extra_has_deletion"] = torch.clip(
            protein["extra_deletion_matrix"], 0.0, 1.0
        )
        protein["extra_deletion_value"] = torch.atan(
            protein["extra_deletion_matrix"] / 3.0
        ) * (2.0 / np.pi)

    protein["msa_feat"] = torch.cat(msa_feat, dim=-1)
    protein["target_feat"] = torch.cat(target_feat, dim=-1)
    return protein


def np_make_sequence_features(sequence: str):
    """Construct a feature dict of sequence features."""
    num_res = len(sequence)
    features = {}
    features["aaseq"] = sequence
    features["aatype"] = rc.sequence_to_onehot(
        sequence=sequence,
        mapping=rc.restype_order_with_x,
        map_unknown_to_x=True,
    )
    features["between_segment_residues"] = np.zeros((num_res,), dtype=np.int32)
    features["residue_index"] = np.array(range(num_res), dtype=np.int32)
    features["seq_length"] = np.array([num_res] * num_res, dtype=np.int32)
    features["sequence"] = np.array(
        [sequence.encode("utf-8")], dtype=object
    )
    return features


def np_make_msa_features(msas, deletion_matrices):
    """Constructs a feature dict of MSA features."""
    if not msas:
        raise ValueError("At least one MSA must be provided.")

    int_msa = []
    deletion_matrix = []
    seen_sequences = set()
    for msa_index, msa in enumerate(msas):
        if not msa:
            raise ValueError(
                f"MSA {msa_index} must contain at least one sequence."
            )
        for sequence_index, sequence in enumerate(msa):
            if sequence in seen_sequences:
                continue
            seen_sequences.add(sequence)
            int_msa.append(
                [rc.HHBLITS_AA_TO_ID[res] for res in sequence]
            )
            deletion_matrix.append(deletion_matrices[msa_index][sequence_index])

    num_res = len(msas[0][0])
    num_alignments = len(int_msa)
    features = {}
    features["deletion_matrix_int"] = np.array(deletion_matrix, dtype=np.int32)
    features["msa"] = np.array(int_msa, dtype=np.int32)
    features["num_alignments"] = np.array(
        [num_alignments] * num_res, dtype=np.int32
    )
    return features

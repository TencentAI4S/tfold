# Copyright (c) 2024, Tencent Inc. All rights reserved.
# Data: 2024/5/29 14:48
import copy
import itertools

import numpy as np
import torch

from tfold.utils.tensor import batched_gather
from . import processing_msa
from .utils import map_fn, compose, curry1
from .. import residue_constants as rc

NUM_RES = "num residues placeholder"
NUM_MSA_SEQ = "msa placeholder"
NUM_EXTRA_SEQ = "extra msa placeholder"
NUM_TEMPLATES = "num templates placeholder"


def make_pseudo_beta(aatype, atom37_positions, atom37_mask=None):
    """Create pseudo beta features for distogram loss

    Args:
        aatype: [*, seq_len],
        atom37_positions: [*, seq_len, 37, 3]
        atom37_mask: [*, seq_len, 37]

    Returns:
        pseudo_beta: [*, seq_len, 3]
        pseudo_beta_mask: [*, seq_len]
    """
    is_gly = torch.eq(aatype, rc.restype_order["G"])
    ca_idx = rc.atom_order["CA"]
    cb_idx = rc.atom_order["CB"]
    pseudo_beta = torch.where(
        torch.tile(is_gly[..., None],
                   [1] * len(is_gly.shape) + [3]),
        atom37_positions[..., ca_idx, :],
        atom37_positions[..., cb_idx, :],
    )

    if atom37_mask is not None:
        pseudo_beta_mask = torch.where(
            is_gly, atom37_mask[..., ca_idx], atom37_mask[..., cb_idx]
        )
        return pseudo_beta, pseudo_beta_mask

    return pseudo_beta


def cast_to_64bit_ints(protein):
    # We keep all ints as int64
    for k, v in protein.items():
        if v.dtype == torch.int32:
            protein[k] = v.type(torch.int64)

    return protein


def make_seq_mask(protein):
    protein["seq_mask"] = torch.ones(
        protein["aatype"].shape, dtype=torch.float32
    )
    return protein


def make_aatype(aaseq, device=None, table=rc.restype_order_with_x):
    if isinstance(aaseq, (list, tuple)):
        return torch.stack([make_aatype(seq, device, table=table) for seq in aaseq], dim=0)

    seq_len = len(aaseq)
    aatype = torch.zeros((seq_len,), dtype=torch.long, device=device)
    for i, aa in enumerate(aaseq):
        aatype[i] = table[aa]

    return aatype


def make_atom37_mask(aatype, dtype=None):
    """make atom37 exist mask"""
    device = aatype.device
    dtype = torch.float32 if dtype is None else dtype
    restype_atom37_mask = torch.zeros(
        [21, 37], dtype=dtype, device=device
    )
    for restype, restype_letter in enumerate(rc.restypes):
        restype_name = rc.restype_1to3[restype_letter]
        atom_names = rc.residue_atoms[restype_name]
        for atom_name in atom_names:
            atom_idx = rc.atom_order[atom_name]
            restype_atom37_mask[restype, atom_idx] = 1

    return restype_atom37_mask[aatype]


def atom14_to_atom37_positions(aatype, atom14_positions, atom14_mask=None):
    assert atom14_positions.shape[-2:] == (
        14, 3), f"expect atom14 positions shape but get shape: {atom14_positions.shape}"
    if atom14_mask is not None:
        assert atom14_positions.shape[:2] == atom14_mask.shape

    residx_atom37_to_atom14 = []
    for rt in rc.restypes:
        atom_names = rc.restype_name_to_atom14_names[rc.restype_1to3[rt]]
        atom_name_to_idx14 = {name: i for i, name in enumerate(atom_names)}
        residx_atom37_to_atom14.append(
            [
                (atom_name_to_idx14[name] if name in atom_name_to_idx14 else 0)
                for name in rc.atom_types
            ]
        )
    residx_atom37_to_atom14.append([0] * 37)
    residx_atom37_to_atom14 = torch.tensor(
        residx_atom37_to_atom14,
        dtype=torch.int32,
        device=aatype.device,
    )[aatype]

    atom37_atom_exists = make_atom37_mask(aatype)
    atom37_positions = atom37_atom_exists[..., None] * batched_gather(
        atom14_positions,
        residx_atom37_to_atom14,
        dim=-2,
        no_batch_dims=len(atom14_positions.shape[:-2]),
    )

    # validness masks for specified residue(s) & atom(s)
    if atom14_mask is not None:
        atom37_mask = atom37_atom_exists * batched_gather(
            atom14_mask,
            residx_atom37_to_atom14,
            dim=-1,
            no_batch_dims=len(atom14_mask.shape[:-1]))
        return atom37_positions, atom37_mask

    return atom37_positions


def make_atom14_masks(protein):
    """Construct denser atom positions (14 dimensions instead of 37)."""
    restype_atom14_to_atom37 = []
    restype_atom37_to_atom14 = []
    restype_atom14_mask = []

    for rt in rc.restypes:
        atom_names = rc.restype_name_to_atom14_names[rc.restype_1to3[rt]]
        restype_atom14_to_atom37.append(
            [(rc.atom_order[name] if name else 0) for name in atom_names]
        )
        atom_name_to_idx14 = {name: i for i, name in enumerate(atom_names)}
        restype_atom37_to_atom14.append(
            [
                (atom_name_to_idx14[name] if name in atom_name_to_idx14 else 0)
                for name in rc.atom_types
            ]
        )

        restype_atom14_mask.append(
            [(1.0 if name else 0.0) for name in atom_names]
        )

    # Add dummy mapping for restype 'UNK'
    restype_atom14_to_atom37.append([0] * 14)
    restype_atom37_to_atom14.append([0] * 37)
    restype_atom14_mask.append([0.0] * 14)

    restype_atom14_to_atom37 = torch.tensor(
        restype_atom14_to_atom37,
        dtype=torch.int32,
        device=protein["aatype"].device,
    )
    restype_atom37_to_atom14 = torch.tensor(
        restype_atom37_to_atom14,
        dtype=torch.int32,
        device=protein["aatype"].device,
    )
    restype_atom14_mask = torch.tensor(
        restype_atom14_mask,
        dtype=torch.float32,
        device=protein["aatype"].device,
    )
    protein_aatype = protein['aatype'].to(torch.long)

    # create the mapping for (residx, atom14) --> atom37, i.e. an array
    # with shape (num_res, 14) containing the atom37 indices for this protein
    residx_atom14_to_atom37 = restype_atom14_to_atom37[protein_aatype]
    residx_atom14_mask = restype_atom14_mask[protein_aatype]

    protein["atom14_atom_exists"] = residx_atom14_mask
    protein["residx_atom14_to_atom37"] = residx_atom14_to_atom37.long()

    # create the gather indices for mapping back
    residx_atom37_to_atom14 = restype_atom37_to_atom14[protein_aatype]
    protein["residx_atom37_to_atom14"] = residx_atom37_to_atom14.long()

    # create the corresponding mask
    restype_atom37_mask = torch.zeros(
        [21, 37], dtype=torch.float32, device=protein["aatype"].device
    )
    for restype, restype_letter in enumerate(rc.restypes):
        restype_name = rc.restype_1to3[restype_letter]
        atom_names = rc.residue_atoms[restype_name]
        for atom_name in atom_names:
            atom_type = rc.atom_order[atom_name]
            restype_atom37_mask[restype, atom_type] = 1

    residx_atom37_mask = restype_atom37_mask[protein_aatype]
    protein["atom37_atom_exists"] = residx_atom37_mask

    return protein


@curry1
def select_feat(protein, feature_list):
    return {k: v for k, v in protein.items() if k in feature_list}


@curry1
def make_fixed_size(
        protein,
        shape_schema,
        msa_cluster_size,
        extra_msa_size,
        num_res=0,
        num_templates=0,
):
    """Guess at the MSA and sequence dimension to make fixed size."""
    pad_size_map = {
        NUM_RES: num_res,
        NUM_MSA_SEQ: msa_cluster_size,
        NUM_EXTRA_SEQ: extra_msa_size,
        NUM_TEMPLATES: num_templates,
    }

    for k, v in protein.items():
        # Don't transfer this to the accelerator.
        if k == "extra_cluster_assignment":
            continue
        shape = list(v.shape)
        schema = shape_schema[k]
        msg = "Rank mismatch between shape and shape schema for"
        assert len(shape) == len(schema), f"{msg} {k}: {shape} vs {schema}"
        pad_size = [
            pad_size_map.get(s2, None) or s1 for (s1, s2) in zip(shape, schema)
        ]

        padding = [(0, p - v.shape[i]) for i, p in enumerate(pad_size)]
        padding.reverse()
        padding = list(itertools.chain(*padding))
        if padding:
            protein[k] = torch.nn.functional.pad(v, padding)
            protein[k] = torch.reshape(protein[k], pad_size)

    return protein


def ensembled_transform_fns(common_cfg, mode_cfg, ensemble_seed):
    """Input pipeline data transformers that can be ensembled and averaged."""
    transforms = []
    pad_msa_clusters = mode_cfg.max_msa_clusters
    max_msa_clusters = pad_msa_clusters
    max_extra_msa = mode_cfg.max_extra_msa

    msa_seed = None
    if not common_cfg.resample_msa_in_recycling:
        msa_seed = ensemble_seed

    transforms.append(
        processing_msa.sample_msa(
            max_msa_clusters,
            keep_extra=True,
            seed=msa_seed,
        )
    )

    if "masked_msa" in common_cfg:
        # Masked MSA should come *before* MSA clustering so that
        # the clustering and full MSA profile do not leak information about
        # the masked locations and secret corrupted locations.
        transforms.append(
            processing_msa.make_masked_msa(
                common_cfg.masked_msa, mode_cfg.masked_msa_replace_fraction
            )
        )

    if common_cfg.msa_cluster_features:
        transforms.append(processing_msa.nearest_neighbor_clusters())
        transforms.append(processing_msa.summarize_clusters())

    # Crop after creating the cluster profiles.
    if max_extra_msa:
        transforms.append(processing_msa.crop_extra_msa(max_extra_msa))
    else:
        transforms.append(processing_msa.delete_extra_msa)

    transforms.append(processing_msa.make_msa_feat())

    crop_feats = dict(common_cfg.feat)
    transforms.append(select_feat(list(crop_feats)))
    transforms.append(
        make_fixed_size(
            crop_feats,
            pad_msa_clusters,
            mode_cfg.max_extra_msa,
            mode_cfg.crop_size,
            mode_cfg.max_templates,
        )
    )

    return transforms


def squeeze_features(protein):
    """Remove singleton and repeated dimensions in protein features."""
    protein["aatype"] = torch.argmax(protein["aatype"], dim=-1)
    for k in [
        "domain_name",
        "msa",
        "num_alignments",
        "seq_length",
        "sequence",
        "superfamily",
        "deletion_matrix",
        "resolution",
        "between_segment_residues",
        "residue_index",
        "template_all_atom_mask",
    ]:
        if k in protein:
            final_dim = protein[k].shape[-1]
            if isinstance(final_dim, int) and final_dim == 1:
                if torch.is_tensor(protein[k]):
                    protein[k] = torch.squeeze(protein[k], dim=-1)
                else:
                    protein[k] = np.squeeze(protein[k], axis=-1)

    for k in ["seq_length", "num_alignments"]:
        if k in protein:
            protein[k] = protein[k][0]

    return protein


def nonensembled_transform_fns():
    """Input pipeline data transformers that are not ensembled."""
    transforms = [
        cast_to_64bit_ints,
        processing_msa.correct_msa_restypes,
        squeeze_features,
        make_seq_mask,
        processing_msa.make_msa_mask,
        processing_msa.make_hhblits_profile,
    ]
    transforms.extend(
        [
            make_atom14_masks,
        ]
    )

    return transforms


def process_tensors_from_config(tensors, common_cfg, mode_cfg, ensemble_seed=None):
    """Based on the config, apply filters and transformations to the data."""
    ensemble_seed = ensemble_seed if ensemble_seed is not None else torch.Generator().seed()

    def wrap_ensemble_fn(data, i):
        """Function to be mapped over the ensemble dimension."""
        d = data.copy()
        fns = ensembled_transform_fns(
            common_cfg,
            mode_cfg,
            ensemble_seed,
        )
        fn = compose(fns)
        d["ensemble_index"] = i
        return fn(d)

    nonensembled = nonensembled_transform_fns()
    tensors = compose(nonensembled)(tensors)

    if "no_recycling_iters" in tensors:
        num_recycling = int(tensors["no_recycling_iters"])
    else:
        num_recycling = common_cfg.max_recycling_iters

    tensors = map_fn(
        lambda x: wrap_ensemble_fn(tensors, x), torch.arange(num_recycling + 1)
    )

    return tensors


def np_example_to_predict_features(np_example, config, seed=42):
    np_example = dict(np_example)
    num_res = len(np_example["aatype"])
    cfg = copy.deepcopy(config)
    with cfg.unlocked():
        cfg.predict.crop_size = num_res

    feature_names = cfg.common.unsupervised_features
    if cfg.common.use_templates:
        feature_names += cfg.common.template_features

    if "deletion_matrix_int" in np_example:
        np_example["deletion_matrix"] = np_example.pop(
            "deletion_matrix_int"
        ).astype(np.float32)

    tensor_dict = {
        k: torch.tensor(v) for k, v in np_example.items() if k in feature_names
    }

    features = process_tensors_from_config(tensor_dict, cfg.common, cfg.predict, seed)

    return {k: v for k, v in features.items()}

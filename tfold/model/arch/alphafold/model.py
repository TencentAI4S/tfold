# Copyright (c) 2024, Tencent Inc. All rights reserved.
# Data: 2024/5/29 15:05
import torch

from tfold.model.module.evoformer import EvoformerStack
from tfold.protein import residue_constants as rc, data_transform
from tfold.utils.tensor import tensor_tree_map, batched_gather
from .auxiliary_head import AuxiliaryHeads
from .module.embedding import TemplateEmbedding, ExtraMSAEmbedder, ExtraMSAStack, RecyclingEmbedding, InputEmbedding
from .module.structure_module import StructureModule
from .. import BaseModel


def atom14_to_atom37(atom14, batch):
    atom37_data = batched_gather(
        atom14,
        batch["residx_atom37_to_atom14"],
        dim=-2,
        no_batch_dims=len(atom14.shape[:-2]),
    )

    atom37_data = atom37_data * batch["atom37_atom_exists"][..., None]

    return atom37_data


class AlphaFold(BaseModel):
    """
    Args:
        config: A dict-like config object
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.model
        self.globals = cfg.globals
        self.max_recycling_iters = cfg.data.common.max_recycling_iters
        self.recycling_embedder = RecyclingEmbedding(**self.cfg.recycling_embedder)
        self.input_embedder = InputEmbedding(**self.cfg.input_embedder)
        self._build_extra_embedding()
        self.evoformer = EvoformerStack(**self.cfg.evoformer_stack)
        self.structure_module = StructureModule(**self.cfg.structure_module)
        self.aux_heads = AuxiliaryHeads(self.cfg.heads)

    def _build_extra_embedding(self):
        self.template_config = self.cfg.template
        self.extra_msa_config = self.cfg.extra_msa
        if self.template_config.enabled:
            self.template_embedder = TemplateEmbedding(self.template_config)

        if self.extra_msa_config.enabled:
            self.extra_msa_embedder = ExtraMSAEmbedder(**self.extra_msa_config.extra_msa_embedder)
            self.extra_msa_stack = ExtraMSAStack(**self.extra_msa_config.extra_msa_stack)

    def _embed_templates(self, batch, z, pair_mask, chunk_size=None):
        template_embeds = self.template_embedder(
            batch,
            z,
            pair_mask,
            chunk_size=chunk_size
        )
        return template_embeds

    def _input_embedding_iml(self, feats, prevs):
        # Grab some data about the input
        aatype = feats["aatype"]
        batch_dims = aatype.shape[:-1]
        seq_len = aatype.shape[-1]
        dtype = self.dtype
        m, z = self.input_embedder(target_feat=feats["target_feat"],
                                   msa_feat=feats["msa_feat"],
                                   residue_index=feats["residue_index"])
        m_1_prev, z_prev, x_prev = prevs
        # lazy initialize the recycling embeddings
        if None in [m_1_prev, z_prev, x_prev]:
            m_1_prev = m.new_zeros((*batch_dims, seq_len, self.cfg.input_embedder.c_m))
            z_prev = z.new_zeros((*batch_dims, seq_len, seq_len, self.cfg.input_embedder.c_z))
            x_prev = z.new_zeros((*batch_dims, seq_len, rc.num_atom_types, 3))

        x_prev = data_transform.make_pseudo_beta(aatype, x_prev).to(dtype=dtype)
        m_1_prev_emb, z_prev_emb = self.recycling_embedder(m_1_prev, z_prev, x_prev)
        # [*, S_c, N, C_m]
        m[..., 0, :, :] += m_1_prev_emb
        z = z + z_prev_emb
        return m, z

    def _extra_embedding_impl(self, m, z, feats, chunk_size=None):
        seq_mask = feats.get("seq_mask", None)
        dtype = self.dtype
        # pair mask is None or [bs, seq_len, seq_len]
        pair_mask = None if seq_mask is None else seq_mask[..., None] * seq_mask[..., None, :].to(dtype)
        msa_mask = feats.get("msa_mask", None)
        if msa_mask is not None:
            msa_mask = msa_mask.to(dtype)

        if self.cfg.template.enabled:
            template_embeds = self._embed_templates(feats, z, pair_mask, chunk_size=chunk_size)
            # [*, N, N, C_z]
            z = z + template_embeds.pop("template_pair_embedding")
            if "template_angle_embedding" in template_embeds:
                # [*, S = S_c + S_t, N, C_m]
                m = torch.cat([m, template_embeds["template_angle_embedding"]], dim=-3)
                # [*, S, N]
                torsion_angles_mask = template_embeds["template_mask"]
                msa_mask = torch.cat([msa_mask, torsion_angles_mask[..., 2]], dim=-2)

        if self.cfg.extra_msa.enabled:
            # Embed extra MSA features + merge with pairwise embeddings
            extra_msa_mask = feats.get("extra_msa_mask", None)
            if extra_msa_mask is not None:
                extra_msa_mask = extra_msa_mask.to(dtype=z.dtype)

            # [*, S_e, N, C_e]
            a = self.extra_msa_embedder(
                feats["extra_msa"],
                feats["extra_has_deletion"],
                feats["extra_deletion_value"]
            )
            # [*, N, N, C_z]
            z = self.extra_msa_stack(
                a, z,
                msa_mask=extra_msa_mask,
                pair_mask=pair_mask,
                chunk_size=chunk_size
            )

        return m, z, msa_mask, pair_mask

    def _embedding_impl(self, feats, prevs, chunk_size=None):
        m, z = self._input_embedding_iml(feats, prevs)
        m, z, msa_mask, pair_mask = self._extra_embedding_impl(m, z, feats, chunk_size=chunk_size)
        return m, z, msa_mask, pair_mask

    def iteration(self, feats, prevs):
        aatype = feats["aatype"]
        num_seqs = feats["msa_feat"].shape[-3]
        m, z, msa_mask, pair_mask = self._embedding_impl(feats, prevs, self.globals.chunk_size)
        m, z, s = self.evoformer(m, z,
                                 msa_mask=msa_mask,
                                 pair_mask=pair_mask,
                                 chunk_size=self.globals.chunk_size)
        outputs = {}
        outputs["single"] = s
        outputs["pair"] = z
        outputs["msa"] = m[..., :num_seqs, :, :]
        structure = self.structure_module(s, aatype, z=z, mask=feats.get("seq_mask", None))
        outputs["sm_single"] = structure["single"]
        outputs["angles"] = structure["angles"]
        outputs["unnormalized_angles"] = structure["unnormalized_angles"]
        outputs["positions"] = structure.pop("positions")
        # outputs["final_atom_positions"] = data_transform.atom14_to_atom37_positions(
        #     aatype,
        #     outputs["positions"][-1]
        # )
        outputs["final_atom_positions"] = atom14_to_atom37(outputs["positions"][-1], feats)
        outputs["final_atom_mask"] = feats["atom37_atom_exists"]

        outputs["final_frames"] = structure["frames"][-1]
        # Save embeddings for use during the next recycling iteration
        # [*, N, C_m]
        m_1_prev = m[..., 0, :, :]
        # [*, N, N, C_z]
        z_prev = z
        # [*, N, 3]
        x_prev = outputs["final_atom_positions"]
        return outputs, m_1_prev, z_prev, x_prev

    def forward(self, batch):
        """
        Args:
            The final dimension is the number of recycling iterations.
            Features (without the recycling dimension):
                "aatype" ([*, N_res]): aa type id
                "target_feat" ([*, N_res, C_tf])
                    One-hot encoding of the target sequence. C_tf is
                    config.model.input_embedder.tf_dim.
                "residue_index" ([*, N_res])
                    Tensor whose final dimension consists of
                    consecutive indices from 0 to N_res.
                "msa_feat" ([*, N_seq, N_res, C_msa])
                    MSA features, constructed as in the supplement.
                    C_msa is config.model.input_embedder.msa_dim.
                "seq_mask" ([*, N_res]) 1-D sequence mask
                "msa_mask" ([*, N_seq, N_res]) MSA mask
                "pair_mask" ([*, N_res, N_res]) 2-D pair mask
                "extra_msa_mask" ([*, N_extra, N_res]) Extra MSA mask
                "template_mask" ([*, N_templ])
                    Template mask (on the level of templates, not
                    residues)
                "template_aatype" ([*, N_templ, N_res])
                    Tensor of template residue indices (indices greater
                    than 19 are clamped to 20 (Unknown))
                "template_all_atom_positions"
                    ([*, N_templ, N_res, 37, 3])
                    Template atom coordinates in atom37 format
                "template_all_atom_mask" ([*, N_templ, N_res, 37])
                    Template atom coordinate mask
                "template_pseudo_beta" ([*, N_templ, N_res, 3])
                    Positions of template carbon "pseudo-beta" atoms
                    (i.e. C_beta for all residues but glycine, for
                    for which C_alpha is used instead)
                "template_pseudo_beta_mask" ([*, N_templ, N_res])
                    Pseudo-beta mask
        """
        # Initialize recycling embeddings
        m_1_prev, z_prev, x_prev = None, None, None
        prevs = [m_1_prev, z_prev, x_prev]
        is_grad_enabled = torch.is_grad_enabled()
        num_cycles = batch["aatype"].shape[-1]
        for cycle_idx in range(num_cycles):
            # Select the features for the current recycling cycle
            fetch_cur_batch = lambda t: t[..., cycle_idx]
            feats = tensor_tree_map(fetch_cur_batch, batch)
            # Enable grad iff we're training and it's the final recycling layer
            is_final_iter = cycle_idx == (num_cycles - 1)
            with torch.set_grad_enabled(is_grad_enabled and is_final_iter):
                # Run the next iteration of the model
                outputs, m_1_prev, z_prev, x_prev = self.iteration(feats, prevs)
                if not is_final_iter:
                    del outputs
                    prevs = [m_1_prev, z_prev, x_prev]
                    del m_1_prev, z_prev, x_prev

        outputs.update(self.aux_heads(outputs))
        return outputs

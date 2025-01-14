# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
import torch
from torch import nn

from tfold.config import configurable
from tfold.model.layer import (RelativePositionEmbedding, ChainRelativePositionEmbedding, MultimerPositionEmebedding,
                               LearnableResidueEmbedding)
from tfold.model.module.evoformer import EvoformerStackSS
from .module import PairPredictor, RecyclingEmbedding, StructureModule
from ..base_model import BaseModel
from ...build import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class ComplexStructureModel(BaseModel):
    """protein complex structure prediction model.

    Args:
        c_s: number of dimensions in single features (D_s)
        c_z: number of dimensions in pair features (D_p)
        init_c_s: number of dimensions in initial single features (D_si)
        init_c_z: number of dimensions in initial pair features (D_pi)
        position_embedding_dim: number of dimensions in positional encodings
        num_2d_layers: number of evoformer layers
        tm_enabled: predict TMscore or not
    """

    @classmethod
    def from_config(cls, cfg):
        return {
            'c_s': cfg.model.c_s,
            'c_z': cfg.model.c_z,
            'init_c_s': cfg.model.init_c_s,
            'init_c_z': cfg.model.init_c_z,
            'position_embedding_dim': cfg.model.position_embedding_dim,
            'num_2d_layers': cfg.model.num_2d_layers,
            'num_3d_layers': cfg.model.num_3d_layers,
            'use_residue_embedding': cfg.model.use_residue_embedding,
            'tm_enabled': cfg.model.tm_enabled,
            'num_recycles': cfg.model.num_recycles,
            'num_3d_recycles': cfg.model.num_3d_recycles
        }

    @configurable
    def __init__(
            self,
            c_s=384,
            c_z=256,
            init_c_s=None,
            init_c_z=None,
            position_embedding_dim=64,
            num_2d_layers=16,
            num_3d_layers=8,
            use_residue_embedding=False,
            tm_enabled=True,
            num_recycles=1,
            num_3d_recycles=8
    ):
        super().__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.init_c_s = init_c_s or c_s
        self.init_c_z = init_c_z or c_z

        self.position_embedding_dim = position_embedding_dim
        self.num_2d_layers = num_2d_layers
        self.num_3d_layers = num_3d_layers
        self.num_recycles = num_recycles
        self.num_3d_recycles = num_3d_recycles
        self.tm_enabled = tm_enabled

        self.posi_encoder = MultimerPositionEmebedding(self.position_embedding_dim)
        self.rpe_encoder = RelativePositionEmbedding(self.c_z)
        self.crpe_encoder = ChainRelativePositionEmbedding(self.c_z)

        self.use_residue_embedding = use_residue_embedding
        if self.use_residue_embedding:
            self.resd_encoder = LearnableResidueEmbedding(self.c_s, self.c_z)

        self.net = nn.ModuleDict({
            'linear-si': nn.Linear(self.init_c_s + self.position_embedding_dim, self.c_s),
            'linear-pi': nn.Linear(self.init_c_z, self.c_z),
            'evoformer': EvoformerStackSS(
                self.c_s,
                self.c_z,
                num_layers=self.num_2d_layers
            ),
            'af2_smod': StructureModule(
                num_layers=self.num_3d_layers,
                c_s=self.c_s,
                c_z=self.c_z,
                n_dims_encd=self.position_embedding_dim,
                tmsc_pred=self.tm_enabled
            ),
            'rc_embed': RecyclingEmbedding(self.c_s, self.c_z),
            'da_pred': PairPredictor(
                self.c_z,
                bins=[37, 25, 25, 25],
                activation='relu_squared'
            )
        })

    def forward(self,
                sequences,
                s_init,
                z_init,
                asym_id=None,
                num_recycles=None,
                num_3d_recycles=None,
                chunk_size=None):
        """
        Args:
            num_recycles: number of global (2D + 3D) recycling iterations
            num_3d_recycles: number of recycling iterations in structure moudule

        Returns:
            outputs: dict of model predictions
        """
        if s_init.dtype != self.dtype:
            s_init = s_init.to(self.dtype)
            z_init = z_init.to(self.dtype)

        num_recycles = self.num_recycles if num_recycles is None else num_recycles
        num_3d_recycles = self.num_3d_recycles if num_3d_recycles is None else num_3d_recycles
        seq_lens = [len(x) for x in sequences]
        aa_seq = "".join(sequences)
        penc_tns = self.posi_encoder(seq_lens, dtype=self.dtype, device=self.device)[None]
        s_init = torch.cat([s_init, penc_tns], dim=2)
        # initial linear mapping
        sfea_tns = self.net['linear-si'](s_init)
        pfea_tns = self.net['linear-pi'](z_init)

        # update single & pair features w/ residue encodings
        if self.use_residue_embedding:
            sfea_tns_renc, pfea_tns_renc = self.resd_encoder(aa_seq)
            sfea_tns += sfea_tns_renc
            pfea_tns += pfea_tns_renc

        # update pair feature w/ relative positional encodings
        pfea_tns += self.rpe_encoder(aa_seq)

        # update pair features w/ chain relative positional encodings
        if asym_id is not None:
            pfea_tns += self.crpe_encoder(seq_lens, asym_id)

        rc_inputs = None
        cord_tns = None
        for cycle_id in range(num_recycles):
            requires_grad = (self.training and (cycle_id == num_recycles - 1))
            with torch.set_grad_enabled(requires_grad):
                if rc_inputs is not None:
                    sfea_tns_ext, pfea_tns = self.net['rc_embed'](
                        aa_seq, sfea_tns.unsqueeze(dim=1), pfea_tns, rc_inputs)
                    sfea_tns = sfea_tns_ext.squeeze(dim=1)

                sfea_tns, pfea_tns = self.net['evoformer'](sfea_tns, pfea_tns, chunk_size=chunk_size)
                # inter-residue distance & angle predictions
                logt_tns_cb, logt_tns_om, logt_tns_th, logt_tns_ph = self.net['da_pred'](pfea_tns)
                outputs_2d = {'cb': logt_tns_cb, 'om': logt_tns_om, 'th': logt_tns_th, 'ph': logt_tns_ph}

                params_list, plddt_list, cord_list, fram_tns_sc, tmsc_dict = self.net['af2_smod'](
                    aa_seq, sfea_tns, pfea_tns, penc_tns, num_3d_recycles, asym_id=asym_id)
                cord_tns = cord_list[-1]
                outputs_3d = {
                    'params': params_list, 'plddt': plddt_list, 'cord': cord_list,
                    'fram_sc': fram_tns_sc, 'tmsc_dict': tmsc_dict,
                }

                # update recycling inputs for the next iteration
                rc_inputs = {
                    'sfea': sfea_tns,
                    'pfea': pfea_tns,
                    'cord': cord_tns
                }

        outputs = {
            'sfea': sfea_tns,
            'pfea': pfea_tns,
            '2d': outputs_2d,
            '3d': outputs_3d,
            'cord': cord_tns
        }

        return outputs

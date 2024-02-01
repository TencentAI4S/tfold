# -*- coding: utf-8 -*-
# Copyright (c) 2023, Tencent Inc. All rights reserved.
import torch
from torch import nn

from tfold.config import configurable, get_config
from tfold.protein.utils import get_chain_ids
from ..build import MODEL_REGISTRY
from ..layer import (RelativePositionEmbedding, ChainRelativePositionEmbedding, MultimerPositionEmebedding,
                     LearnableResidueEmbedding)
from ..module import PairPredictor, RecyclingEmbedding, EvoformerStackSS, StructureModule


@MODEL_REGISTRY.register()
class ComplexStructureModel(nn.Module):
    """The antibody structure prediction model.

    Notes:
        This model can also be viewed as a general model for protein complex structure prediction.
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
            'num_2d_recycles': cfg.model.num_2d_recycles,
            'num_3d_recycles': cfg.model.num_3d_recycles
        }

    @configurable
    def __init__(
            self,
            c_s=384,
            c_z=256,
            init_c_s=-1,
            init_c_z=-1,
            position_embedding_dim=64,
            num_2d_layers=16,
            num_3d_layers=8,
            use_residue_embedding=False,
            tm_enabled=True,
            num_recycles=1,
            num_2d_recycles=1,
            num_3d_recycles=8
    ):
        super().__init__()
        self.init_c_s = init_c_s
        self.init_c_z = init_c_z
        self.c_s = c_s
        self.c_z = c_z
        self.position_embedding_dim = position_embedding_dim
        self.num_2d_layers = num_2d_layers
        self.num_3d_layers = num_3d_layers
        self.num_recycles = num_recycles
        self.num_2d_recycles = num_2d_recycles
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
                tmsc_pred=self.tm_enabled,
            ),
            'rc_embed': RecyclingEmbedding(self.c_s, self.c_z),
            # PairPredictor (auxiliary predictions for inter-residue geometries)
            'da_pred': PairPredictor(
                self.c_z,
                bins=[37, 25, 25, 25],
                activation='relu_squared'
            )
        })

    @classmethod
    def restore(cls, path):
        state = torch.load(path, map_location='cpu')
        config = get_config()
        config.update(state['config'])
        model = cls(config)
        model.load_state_dict(state['model'])

        return model

    def forward(self,
                inputs,
                num_recycles=None,
                num_2d_recycles=None,
                num_3d_recycles=None,
                chunk_size=None):
        """
        Args:
            inputs: dict of input tensors (amino-acid sequence & initial single/pair features)

        Returns:
            outputs: dict of model predictions

        Notes:
        * The input dict is organized as below (unused data entries are omitted here):
          > chn1:
            > base:
              > seq: first chain's amino-acid sequence of length L1
            > feat: (only used for monomer inputs)
              > sfea: initial single features of size 1 x L1 x D_s
              > pfea: initial pair features of size 1 x L1 x L1 x D_p
          > chn2: (optional; only available for multimer inputs)
            > base:
              > seq: second chain's amino-acid sequence of length L2
          > chn1-chn2: (optional; only available for multimer inputs)
            > base:
              > seq: complex's amino-acid sequence of length Lc = (L1 + L2) (no linker)
            > asym_id: asymmetric ID of length Lc
            > feat: (only used for multimer inputs)
              > sfea: initial single features of size 1 x L_c x D_s
              > pfea: initial pair features of size 1 x L_c x L_c x D_p
        * The config dict is organized as below:
          > n_recys_gb: number of global (2D + 3D) recycling iterations
          > n_recys_2d: number of 2D-only recycling iterations (EvoformerBlockSS)
          > n_recys_3d: number of 3D-only recycling iterations (AF2SMod)
        """
        num_recycles = self.num_recycles if num_recycles is None else num_recycles
        num_2d_recycles = self.num_2d_recycles if num_2d_recycles is None else num_2d_recycles
        num_3d_recycles = self.num_3d_recycles if num_3d_recycles is None else num_3d_recycles

        device = self.net['linear-si'].weight.device
        # determine the input type (monomer OR multimer)
        chain_ids = get_chain_ids(inputs)
        aa_seq_list = [inputs[x]['base']['seq'] for x in chain_ids]
        n_resds_list = [len(x) for x in aa_seq_list]
        if len(chain_ids) == 1:
            is_mono = True
            chain_id = chain_ids[0]
            aa_seq = aa_seq_list[0]
        elif len(chain_ids) == 2:
            is_mono = False
            chain_id = '-'.join(chain_ids)
            aa_seq = ''.join(aa_seq_list)
            assert (chain_id in inputs) and (aa_seq == inputs[chain_id]['base']['seq'])
            chain_id_pri, chain_id_sec = chain_ids
        else:
            raise ValueError(f'unexpected list of chain IDs: {chain_ids}')

        # get positional encodings
        chn_infos = list(zip(chain_ids, n_resds_list))
        asym_id = None if is_mono else inputs[chain_id]['asym_id'][0]
        penc_tns = self.posi_encoder(chn_infos)[None].to(device)
        # get initial single & pair features (if PLM is not used)
        sfea_tns_init = inputs[chain_id]['feat']['sfea']
        pfea_tns_init = inputs[chain_id]['feat']['pfea']

        # initial linear mapping
        sfea_tns = self.net['linear-si'](torch.cat([sfea_tns_init, penc_tns], dim=2))
        pfea_tns = self.net['linear-pi'](pfea_tns_init)

        # update single & pair features w/ residue encodings
        if self.use_residue_embedding:
            sfea_tns_renc, pfea_tns_renc = self.resd_encoder(aa_seq)
            sfea_tns += sfea_tns_renc
            pfea_tns += pfea_tns_renc

        # update pair feature w/ relative positional encodings
        pfea_tns += self.rpe_encoder(aa_seq)

        # update pair features w/ chain relative positional encodings
        if not is_mono:
            pfea_tns += self.crpe_encoder(chn_infos, asym_id)

        rc_inputs = None  # will be initialized after the first forward pass
        cord_tns = None  # will be initialized after the first forward pass
        for _ in range(num_recycles):
            # RcEmbedNet
            if rc_inputs is not None:
                sfea_tns_ext, pfea_tns = self.net['rc_embed'](
                    aa_seq, sfea_tns.unsqueeze(dim=1), pfea_tns, rc_inputs)
                sfea_tns = sfea_tns_ext.squeeze(dim=1)

            # Evoformer
            sfea_tns, pfea_tns = self.net['evoformer'](sfea_tns, pfea_tns,
                                                       num_recycles=num_2d_recycles,
                                                       chunk_size=chunk_size)
            # inter-residue distance & angle predictions
            logt_tns_cb, logt_tns_om, logt_tns_th, logt_tns_ph = self.net['da_pred'](pfea_tns)
            outputs_2d = {'cb': logt_tns_cb, 'om': logt_tns_om, 'th': logt_tns_th, 'ph': logt_tns_ph}

            # AF2SMod
            params_list, plddt_list, cord_list, fram_tns_sc, tmsc_dict = self.net['af2_smod'](
                aa_seq, sfea_tns, pfea_tns, penc_tns, num_3d_recycles, asym_id=asym_id)
            cord_tns = cord_list[-1]
            outputs_3d = {
                'params': params_list, 'plddt': plddt_list, 'cord': cord_list,
                'fram_sc': fram_tns_sc, 'tmsc_dict': tmsc_dict,
            }

            # update recycling inputs for the next iteration
            rc_inputs = {
                'sfea': sfea_tns.detach(),
                'pfea': pfea_tns.detach(),
                'cord': cord_tns.detach(),
            }

        # pack output tensors into a dict
        outputs = {
            chain_id: {
                'sfea': sfea_tns,
                'pfea': pfea_tns,
                '2d': outputs_2d,
                '3d': outputs_3d,
                'cord': cord_tns,  # final structure prediction
            },
        }
        if not is_mono:
            outputs_pri, outputs_sec = self._split_outputs(outputs[chain_id], n_resds_list)
            outputs.update({chain_id_pri: outputs_pri, chain_id_sec: outputs_sec})

        return outputs

    @classmethod
    def _split_outputs(cls, outputs_raw, n_resds_list):
        """Split the dict of output tensors into primary & secondary chains.

        Notes:
            Scalar outputs are not re-calculated for primary & secondary chains for simplicity.
        """
        n_resds_pri = n_resds_list[0]

        # basic tensors
        outputs_pri = {
            'sfea': outputs_raw['sfea'][:, :n_resds_pri],
            'pfea': outputs_raw['pfea'][:, :n_resds_pri, :n_resds_pri],
            'cord': outputs_raw['cord'][:n_resds_pri],
        }
        outputs_sec = {
            'sfea': outputs_raw['sfea'][:, n_resds_pri:],
            'pfea': outputs_raw['pfea'][:, n_resds_pri:, n_resds_pri:],
            'cord': outputs_raw['cord'][n_resds_pri:],
        }

        # detailed 2D predictions
        keys = ['cb', 'om', 'th', 'ph']
        outputs_pri['2d'] = {
            k: outputs_raw['2d'][k][:, :, :n_resds_pri, :n_resds_pri] for k in keys}
        outputs_sec['2d'] = {
            k: outputs_raw['2d'][k][:, :, n_resds_pri:, n_resds_pri:] for k in keys}

        # detailed 3D predictions
        keys = ['quat', 'trsl', 'angl', 'quat-u']
        params_list_pri = [{k: x[k][:n_resds_pri] for k in keys} for x in outputs_raw['3d']['params']]
        params_list_sec = [{k: x[k][n_resds_pri:] for k in keys} for x in outputs_raw['3d']['params']]
        keys = ['logit', 'plddt-r']
        plddt_list_pri = [{k: x[k][:n_resds_pri] for k in keys} for x in outputs_raw['3d']['plddt']]
        plddt_list_sec = [{k: x[k][n_resds_pri:] for k in keys} for x in outputs_raw['3d']['plddt']]
        cord_list_pri = [x[:n_resds_pri] for x in outputs_raw['3d']['cord']]
        cord_list_sec = [x[n_resds_pri:] for x in outputs_raw['3d']['cord']]
        fram_tns_sc_pri, fram_tns_sc_sec = \
            torch.split(outputs_raw['3d']['fram_sc'], n_resds_list, dim=0)
        tmsc_dict_pri = {'ptm_logt': outputs_raw['3d']['tmsc_dict']['ptm_logt'][:n_resds_pri, :n_resds_pri]}
        tmsc_dict_sec = {'ptm_logt': outputs_raw['3d']['tmsc_dict']['ptm_logt'][n_resds_pri:, n_resds_pri:]}
        outputs_pri['3d'] = {
            'params': params_list_pri, 'plddt': plddt_list_pri, 'cord': cord_list_pri,
            'fram_sc': fram_tns_sc_pri, 'tmsc_dict': tmsc_dict_pri,
        }
        outputs_sec['3d'] = {
            'params': params_list_sec, 'plddt': plddt_list_sec, 'cord': cord_list_sec,
            'fram_sc': fram_tns_sc_sec, 'tmsc_dict': tmsc_dict_sec,
        }

        return outputs_pri, outputs_sec

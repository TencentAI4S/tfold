# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
import torch
from torch import nn

from tfold.model.layer import Linear, ChainRelativePositionEmbedding, MultimerPositionEmebedding
from tfold.model.module import EvoformerStackSS
from tfold.model.layer.embedding.utils import dist_to_contact
from tfold.protein.data_transform import get_asym_ids
from tfold.protein.prot_constants import RESD_NAMES_1C, RESD_WITH_X
from tfold.protein import ProtStruct
from .module import RecyclingEmbedding, PairPredictor, StructureModule
from .mono_to_multimer import MonoToMultimerSM, MonoToMultimerSS
from ..base_model import BaseModel


class DockingModelSS(BaseModel):
    """The ligand & receptor complex docking model.

    Args:
        ligand_c_s: number of dimensions in single features (D_s)
        ligand_c_z: number of dimensions in pair features (D_p)
        receptor_c_s: number of dimensions in initial single features for receptor
        receptor_c_z: number of dimensions in initial pair features for receptor
        num_2d_layers: number layers of evoformer
        num_3d_layers: number blocks of structure module
    """

    def __init__(
            self,
            ligand_c_s,
            ligand_c_z,
            receptor_c_s,
            receptor_c_z,
            coord_init=None,
            num_2d_layers=16,
            num_3d_layers=8,
            num_recycles=1,
            num_3d_recycles=1
    ):
        super().__init__()
        self.ligand_c_s = ligand_c_s
        self.ligand_c_z = ligand_c_z
        self.receptor_c_s = receptor_c_s
        self.receptor_c_z = receptor_c_z
        self.num_2d_layers = num_2d_layers
        self.num_3d_layers = num_3d_layers
        self.num_recycles = num_recycles
        self.num_3d_recycles = num_3d_recycles
        self.posi_encoder = MultimerPositionEmebedding(64)
        self.crpe_encoder = ChainRelativePositionEmbedding(self.ligand_c_z)
        self.coord_init = coord_init
        self.net = nn.ModuleDict()
        # build module for complex structure prediction
        self.net['mono2mult'] = self.build_fusion_module()
        self.net['evoformer'] = EvoformerStackSS(
            c_s=self.ligand_c_s,
            c_z=self.ligand_c_z,
            num_layers=self.num_2d_layers
        )
        self.net['aa_pred'] = Linear(self.ligand_c_s, len(RESD_NAMES_1C))
        self.net['af2_smod'] = StructureModule(
            c_s=self.ligand_c_s,
            c_z=self.ligand_c_z,
            num_layers=self.num_3d_layers,
            n_dims_encd=64,
            tmsc_pred=True
        )
        self.net['rc_embed'] = RecyclingEmbedding(
            c_m=self.ligand_c_s,
            c_z=self.ligand_c_z
        )
        # auxiliary predictions for inter-residue geometries
        self.net['da_pred'] = PairPredictor(
            self.ligand_c_z,
            bins=[37, 25, 25, 25],
            activation='relu_squared'
        )

    def build_fusion_module(self):
        return MonoToMultimerSS(
            ligand_c_s=self.ligand_c_s,
            ligand_c_z=self.ligand_c_z,
            receptor_c_s=self.receptor_c_s,
            receptor_c_z=self.receptor_c_z
        )

    def init_input_feature(self, inputs):
        ligand_id = inputs["base"]["ligand_id"]
        receptor_id = inputs["base"]["receptor_id"]
        # get ligand initial features
        ligand_feat = inputs[ligand_id]['feat']
        ligand_feat['seq'] = inputs[ligand_id]['base']['seq']
        if 'mask' in inputs[ligand_id]['base']:
            mask = inputs[ligand_id]['base']['mask']
            modified_seq = ''.join(['G' if mask[i] == 1 else ligand_feat['seq'][i]
                                    for i in range(len(ligand_feat['seq']))])
            ligand_feat['seq'] = modified_seq

        # get receptor initial features
        receptor_feat = inputs[receptor_id]['feat']
        receptor_feat['seq'] = inputs[receptor_id]['base']['seq']
        # get initial single & pair features
        s, z = self.net['mono2mult'](ligand_feat, receptor_feat)
        return s, z

    def forward(self,
                inputs,
                num_recycles=None,
                num_3d_recycles=None,
                chunk_size=None):
        """
        Args:
            inputs: dict of input tensors
            num_recycles: number of global (2D + 3D) recycling iterations
            num_3d_recycles: number of 3D-only recycling iterations (AF2SMod)
            chunk_size: chunk_size for inference

        Returns:
            outputs: dict of model predictions

        Notes:
        * The input dict is organized as below (unused data entries are omitted here):
          > ligand:
            > base:
              > seq: ligand's amino-acid sequence of length L1
            > feat:
              > sfea: initial single features of size 1 x L1 x D_s
              > pfea: initial pair features of size 1 x L1 x L1 x D_p
              > cord: initial coordinate features of size 1 x L1 x 14
          > receptor:
            > base:
              > seq: receptor's amino-acid sequence of length L2
            > feat:
              > mfea: initial MSA features of size 1 x M x L2 x D_s
              > pfea: initial pair features of size 1 x L2 x L2 x D_p
              > cord: initial coordinate features of size 1 x L2 x 14
         > ligand:receptor:
            > base:
              > seq: complex's amino-acid sequence of length Lc = (L1 + L2) (no linker)
            > asym_id: asymmetric ID of length Lc
        """
        num_recycles = num_recycles or self.num_recycles
        num_3d_recycles = num_3d_recycles or self.num_3d_recycles
        device = self.device
        ligand_id = inputs["base"]["ligand_id"]
        ligand_feat = inputs[ligand_id]['feat']
        receptor_id = inputs["base"]["receptor_id"]
        receptor_feat = inputs[receptor_id]['feat']
        complex_id = ':'.join([ligand_id, receptor_id])
        lengths = [len(inputs[x]['base']['seq']) for x in [ligand_id, receptor_id]]
        penc_tns = self.posi_encoder(lengths).unsqueeze(dim=0).to(device)
        inputs[complex_id]['feat']['penc'] = penc_tns
        sfea_tns, pfea_tns = self.init_input_feature(inputs)

        # update pair features w/ chain relative positional encodings
        asym_id = inputs[complex_id]['asym_id'][0]
        pfea_tns += self.crpe_encoder(lengths, asym_id)

        rc_inputs = None
        outputs = {}
        for cycle_idx in range(num_recycles):
            # disable gradient computation except for the last recycle
            requires_grad = (self.training and (cycle_idx == num_recycles - 1))
            with torch.set_grad_enabled(requires_grad):
                self.net['evoformer'].requires_grad_(requires_grad)
                if rc_inputs is not None:
                    sfea_tns_ext, pfea_tns = self.net['rc_embed'](
                        inputs[complex_id]['base']['seq'], sfea_tns.unsqueeze(dim=1), pfea_tns, rc_inputs)
                    sfea_tns = sfea_tns_ext.squeeze(dim=1)

                sfea_tns, pfea_tns = self.net['evoformer'](sfea_tns, pfea_tns, chunk_size=chunk_size)

                # MLM prediction head
                if 'mask' in inputs[ligand_id]['base']:
                    logt_tns_aa = self.net['aa_pred'](sfea_tns)
                    logt_tns_aa = logt_tns_aa.permute(0, 2, 1)  # move classification logits to the 2nd dimensions
                    pred_token = torch.argmax(logt_tns_aa, dim=1)
                    inpt_token = torch.LongTensor(
                        [RESD_WITH_X.index(x) for x in inputs[ligand_id]['base']['seq']]).to(device)
                    output_token = torch.where(
                        inputs[ligand_id]['base']['mask'] == 1,
                        pred_token[0][:len(inputs[ligand_id]['base']['seq'])],
                        inpt_token)
                    ligand_seq = ''.join([RESD_NAMES_1C[x] for x in output_token])
                    # update the complex sequence
                    inputs[ligand_id]['base']['seq'] = ligand_seq
                    inputs[complex_id]['base']['seq'] = ligand_seq + inputs[receptor_id]['base']['seq']

                # inter-residue distance & angle predictions
                logt_tns_cb, logt_tns_om, logt_tns_th, logt_tns_ph = self.net['da_pred'](pfea_tns)
                if self.coord_init == "tcr":
                    cord_tns = torch.cat([ligand_feat['cord'], receptor_feat['cord']], dim=0)
                    cmsk_mat = torch.cat(
                        [ProtStruct.get_cmsk_vld(inputs[ligand_id]['base']['seq'], device),
                         ProtStruct.get_cmsk_vld(inputs[receptor_id]['base']['seq'], device)],
                        dim=0
                    )
                    cmsk_mat[len(ligand_feat['seq']):].zero_()
                    cord_tns[len(ligand_feat['seq']):].zero_()
                else:
                    cord_tns = None
                    cmsk_mat = None

                # caclulate the inter-chain contact for actifpTM calculation
                interface_id = get_asym_ids(
                    [inputs[ligand_id]['base']['seq'], inputs[receptor_id]['base']['seq']]).to(device)
                pair_mask = interface_id[:, None] != interface_id[None, :]
                inter_chain_contact = dist_to_contact(logt_tns_cb[0].permute(1, 2, 0), pair_mask)

                params_list, plddt_list, cord_list, fram_tns_sc, tmsc_dict = self.net['af2_smod'](
                    inputs[complex_id]['base']['seq'],
                    sfea_tns,
                    pfea_tns,
                    penc_tns,
                    asym_id=asym_id,
                    cord_tns=cord_tns,
                    cmsk_mat=cmsk_mat,
                    inter_chain_contact=inter_chain_contact,
                )

                # pack all the outputs into a dict
                outputs = {
                    'sfea': sfea_tns,
                    'pfea': pfea_tns,
                    'penc': penc_tns,
                    '1d': {},
                    '2d': {
                        'cb': logt_tns_cb,
                        'om': logt_tns_om,
                        'th': logt_tns_th,
                        'ph': logt_tns_ph,
                    },
                    '3d': {
                        'params': params_list,
                        'plddt': plddt_list,
                        'cord': cord_list,
                        'fram_sc': fram_tns_sc,
                        'tmsc_dict': tmsc_dict,
                    },
                }
                if 'mask' in inputs[ligand_id]['base']:
                    outputs['1d']['pred_tns'] = logt_tns_aa
                    outputs['1d']['seq'] = ligand_seq

                # collect recycling inputs
                rc_inputs = {
                    'sfea': outputs['sfea'].detach(),
                    'pfea': outputs['pfea'].detach(),
                    'cord': outputs['3d']['cord'][-1].detach(),
                }

        return outputs


class DockingModelSM(DockingModelSS):
    """The ligand & receptor complex docking model.

    Workflow:
     * initialize inter-chain pair features
     * (s_abi, p_abi, s_abi, p_agi) => (s_c, p_c) # c: ligand-receptor complex

    Args:
        ligand_c_s: number of dimensions in single features (D_s)
        ligand_c_z: number of dimensions in pair features (D_p)
        receptor_c_s: number of dimensions in initial single features for receptor
        receptor_c_z: number of dimensions in initial pair features for receptor
        num_2d_layers: number layers of evoformer
        num_3d_layers: number blocks of structure module
    """

    def __init__(
            self,
            ligand_c_s,
            ligand_c_z,
            receptor_c_s,
            receptor_c_z,
            num_2d_layers=16,
            num_3d_layers=8,
            num_recycles=1,
            num_3d_recycles=1,
            use_icf=False
    ):
        self.use_icf = use_icf
        super().__init__(ligand_c_s, ligand_c_z, receptor_c_s, receptor_c_z,
                         num_2d_layers=num_2d_layers,
                         num_3d_layers=num_3d_layers,
                         num_recycles=num_recycles,
                         num_3d_recycles=num_3d_recycles)

    def build_fusion_module(self):
        # build module for complex structure prediction
        return MonoToMultimerSM(
            ligand_c_s=self.ligand_c_s,
            ligand_c_z=self.ligand_c_z,
            receptor_c_s=self.receptor_c_s,
            receptor_c_z=self.receptor_c_z,
            use_icf=self.use_icf
        )

    def init_input_feature(self, inputs):
        device = self.device
        ligand_id = inputs["base"]["ligand_id"]
        receptor_id = inputs["base"]["receptor_id"]
        complex_id = ':'.join([ligand_id, receptor_id])
        # get ligand initial features
        ligand_feat = inputs[ligand_id]['feat']
        ligand_feat['seq'] = inputs[ligand_id]['base']['seq']
        if 'mask' in inputs[ligand_id]['base']:
            mask = inputs[ligand_id]['base']['mask']
            modified_seq = ''.join(['G' if mask[i] == 1 else ligand_feat['seq'][i]
                                    for i in range(len(ligand_feat['seq']))])
            ligand_feat['seq'] = modified_seq

        # get receptor initial features
        receptor_feat = {
            'seq': inputs[receptor_id]['base']['seq'],
            'mfea': inputs[receptor_id]['feat']['mfea'][-1].unsqueeze(dim=0).to(ligand_feat['sfea']),
            'pfea': inputs[receptor_id]['feat']['pfea'][-1].unsqueeze(dim=0).to(ligand_feat['pfea']),
            'cord': inputs[receptor_id]['feat']['cord'][-1].to(ligand_feat['cord']),
        }

        # update single & pair features w/ inter-chain features
        if self.use_icf and inputs[complex_id]['feat']['icf'] is not None:
            ic_feat = inputs[complex_id]['feat']['icf'].to(device)
        else:
            ic_feat = None
        # get initial single & pair features
        s, z = self.net['mono2mult'](ligand_feat, receptor_feat, ic_feat)
        return s, z

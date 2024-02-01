# -*- coding: utf-8 -*-
# Copyright (c) 2023, Tencent Inc. All rights reserved.
import logging
import os
from collections import OrderedDict

import torch

from tfold.model import PPIModel, ComplexStructureModel
from tfold.protein import ProtStruct, prot_constants as pc
from tfold.protein.parser import parse_fas_file_mult, PdbParser
from tfold.protein.utils import get_asym_ids
from tfold.utils import to_device


class AbPredictor(torch.nn.Module):
    """The antibody structure predictor."""
    def __init__(self, ppi_path, ab_path):
        super().__init__()
        logging.info('restoring the pre-trained ppi model ...')
        self.plm_featurizer = PPIModel.restore(ppi_path)
        logging.info('restoring the pre-trained tfold ab model ...')
        self.model = ComplexStructureModel.restore(ab_path)
        self.eval()

    @torch.no_grad()
    def forward(self, fas_fpath, output):
        """Run the antibody structure predictor.

        Args:
            fas_fpath: path to the input FASTA file
            output: path to the output PDB file
        """
        prot_id = os.path.basename(fas_fpath).replace('.fasta', '')
        # check early-exit conditions
        if not os.path.exists(fas_fpath):
            logging.warning('[%s] FASTA file not found; skipping ...', prot_id)
            return
        if os.path.exists(output) and (os.stat(output).st_size != 0):
            logging.warning('[%s] PDB file already generated; skipping ...', prot_id)
            return

        # parse the FASTA file and check for non-standard residue(s)
        aa_seq_dict = parse_fas_file_mult(fas_fpath)
        if any(len(set(x) - set(pc.RESD_NAMES_1C)) > 0 for x in aa_seq_dict.values()):
            logging.warning('[%s] non-standard residue(s) detected; skipping ...', prot_id)
            return

        # build an input dict
        aa_seq_dict = OrderedDict({k[-1]: v for k, v in aa_seq_dict.items()})
        inputs = self._build_inputs(prot_id, aa_seq_dict)
        inputs = to_device(inputs, 'cuda:0')

        # get initial single & pair features
        chain_ids = inputs['base']['chain_ids']
        chain_id = '-'.join(chain_ids)
        plm_outputs = self.plm_featurizer(aa_seq_dict)
        inputs[chain_id]['feat']['sfea'] = plm_outputs['sfea']
        inputs[chain_id]['feat']['pfea'] = plm_outputs['pfea']
        outputs = self.model(inputs)
        # export the predicted structure to a PDB file
        prot_data, pred_info = self._build_prot_data(inputs, outputs)
        PdbParser.save_multimer(prot_data, output, pred_info=pred_info)
        logging.info('[%s] PDB file generated', prot_id)

    @classmethod
    def _build_inputs(cls, prot_id, aa_seq_dict):
        """Build an input dict for model inference."""

        # obtain chain IDs
        n_chns = len(aa_seq_dict)
        chain_ids = sorted(list(aa_seq_dict.keys()))
        assert all(x in {'H', 'L'} for x in chain_ids), 'chain ID must be either "H" or "L"'

        # build an input dict
        inputs = {'base': {'id': prot_id, 'chain_ids': chain_ids}}
        if n_chns == 1:
            chain_id = chain_ids[0]
            aa_seq = aa_seq_dict[chain_id]
            inputs[chain_id] = {'base': {'seq': aa_seq}, 'feat': {}}
        elif n_chns == 2:
            chain_id = '-'.join(chain_ids)  # pseudo chain ID for the VH-VL complex
            chain_id_pri, chain_id_sec = chain_ids
            aa_seq_pri, aa_seq_sec = [aa_seq_dict[x] for x in chain_ids]
            inputs[chain_id_pri] = {'base': {'seq': aa_seq_pri}, 'feat': {}}
            inputs[chain_id_sec] = {'base': {'seq': aa_seq_sec}, 'feat': {}}
            inputs[chain_id] = {
                'base': {'seq': aa_seq_pri + aa_seq_sec},
                'asym_id': get_asym_ids([aa_seq_pri, aa_seq_sec]).unsqueeze(dim=0),
                'feat': {},
            }
        else:
            raise ValueError(f'unexpected number of chains: {n_chns}')

        return inputs

    @classmethod
    def _build_prot_data(cls, inputs, outputs):
        """Build a dict of protein structure data."""
        chain_ids = inputs['base']['chain_ids']
        chain_id = '-'.join(chain_ids)
        device = inputs[chain_id]['feat']['sfea'].device

        # build a dict of protein structure data
        pred_info = ''
        if len(chain_ids) == 1:
            aa_seq = inputs[chain_id]['base']['seq']
            prot_data = OrderedDict({
                chain_id: {
                    'seq': aa_seq,
                    'cord': outputs[chain_id]['cord'],
                    'cmsk': ProtStruct.get_cmsk_vld(aa_seq, device),
                    'plddt': outputs[chain_id]['3d']['plddt'][-1]['plddt-r'],
                }
            })
            ptm_val = outputs[chain_id]['3d']['tmsc_dict']['ptm'].item()
            pred_info += f'REMARK 250 Predicted pTM score: {ptm_val:.4f}\n'
        else:  # then the number of chains must be two
            chain_id_pri, chain_id_sec = chain_ids
            aa_seq_pri = inputs[chain_id_pri]['base']['seq']
            aa_seq_sec = inputs[chain_id_sec]['base']['seq']
            prot_data = OrderedDict({
                chain_id_pri: {
                    'seq': aa_seq_pri,
                    'cord': outputs[chain_id_pri]['cord'],
                    'cmsk': ProtStruct.get_cmsk_vld(aa_seq_pri, device),
                    'plddt': outputs[chain_id_pri]['3d']['plddt'][-1]['plddt-r'],
                },
                chain_id_sec: {
                    'seq': aa_seq_sec,
                    'cord': outputs[chain_id_sec]['cord'],
                    'cmsk': ProtStruct.get_cmsk_vld(aa_seq_sec, device),
                    'plddt': outputs[chain_id_sec]['3d']['plddt'][-1]['plddt-r'],
                },
            })
            iptm_val = outputs[chain_id]['3d']['tmsc_dict']['iptm'].item()
            ptm_val = outputs[chain_id]['3d']['tmsc_dict']['ptm'].item()
            pred_info += f'REMARK 250 Predicted ipTM score: {iptm_val:.4f}\n'
            pred_info += f'REMARK 250 Predicted pTM score: {ptm_val:.4f}\n'

        return prot_data, pred_info

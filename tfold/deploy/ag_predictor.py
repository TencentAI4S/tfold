# -*- coding: utf-8 -*-
# Copyright (c) 2023, Tencent Inc. All rights reserved.
import logging
import os
import time
from collections import OrderedDict

import torch
import torch.nn as nn

from tfold.model import AgModel, PPIModel
from tfold.protein import ProtStruct
from tfold.protein.parser import export_fas_file_mult, parse_fas_file_mult, PdbParser, parse_a3m
from tfold.protein import prot_constants as pc
from tfold.protein.utils import get_complex_id, get_asym_ids, get_mlm_masks
from tfold.utils import to_device
from .psp_featurizer import PspFeaturizer


class AgPredictor(nn.Module):
    """The antibody & antigen multimer structure predictor.
    """

    def __init__(self,
                 ppi_path,
                 psp_path,
                 ag_path):
        super().__init__()
        logging.info('restoring the pre-trained tFold-PPI-SeqPT model ...')
        self.plm_featurizer = PPIModel.restore(ppi_path)
        logging.info('restoring the pre-trained AlphaFold2 model ...')
        self.psp_featurizer = PspFeaturizer.restore(psp_path)
        logging.info('restoring the pre-trained tFold-Ag model ...')
        self.model = AgModel.restore(ag_path)
        self.eval()

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, fasta_path, msa_path, pdb_path, icf_path=None):
        """Run the antibody & antigen multimer structure predictor.

        Args:
            fasta_path: path to the input FASTA file
            msa_path: path to the input MSA file
            pdb_path: path to the output PDB file
            icf_path: path to the inter-chain feature files
        """
        prot_id = os.path.basename(fasta_path).replace('.fasta', '')
        if not os.path.exists(fasta_path):
            logging.warning('[%s] FASTA file not found; skipping ...', prot_id)
            return

        if os.path.exists(pdb_path) and (os.stat(pdb_path).st_size != 0):
            logging.warning('[%s] PDB file already generated; skipping ...', prot_id)
            return

        # parse the FASTA and MSA file and check for non-standard residue(s)
        start = time.time()
        aa_seq_dict = parse_fas_file_mult(fasta_path)
        aa_seq_dict = OrderedDict({k[-1]: v for k, v in aa_seq_dict.items()})
        with open(msa_path) as f:
            msa, _ = parse_a3m(f.read())

        if 'A' not in aa_seq_dict:
            aa_seq_dict['A'] = msa[0]

        for chn_id in aa_seq_dict:
            resd_set = set(pc.RESD_NAMES_1C) if chn_id == 'A' else set(pc.RESD_WITH_X)
            if len(set(aa_seq_dict[chn_id]) - resd_set) > 0:
                logging.warning('[%s] non-standard residue(s) detected; skipping ...', prot_id)
                return

        # build an input dict
        inputs = self._build_inputs(prot_id, aa_seq_dict, icf_path)
        inputs = to_device(inputs, self.device)

        # get initial single & pair features
        ligand_id, _ = get_complex_id(inputs)

        aa_seq_dict_ligand = {x: inputs[x]['base']['seq'] for x in ligand_id.split('-')}

        if 'mask' not in inputs[ligand_id]['base']:
            plm_outputs = self.plm_featurizer(aa_seq_dict_ligand)
        else:
            mask_vecs = [inputs[x]['base']['mask'] for x in ligand_id.split('-')]
            plm_outputs = self.plm_featurizer(aa_seq_dict_ligand, mask_prob=1, mask_vecs=mask_vecs)
        inputs[ligand_id]['feat']['sfea'] = plm_outputs['sfea']
        inputs[ligand_id]['feat']['pfea'] = plm_outputs['pfea']

        # get antigen feature w/ af2
        logging.info('[%s] antigen feature predicted using pretrained structure prediction model', prot_id)
        ag_feat, _ = self.psp_featurizer(msa_path)
        inputs['A']['feat'] = ag_feat

        logging.info('start ag model in %.2f second', time.time() - start)
        outputs = self.model(inputs)
        # export the predicted structure to a PDB file
        prot_data, pred_info = self._build_prot_data(inputs, outputs)
        PdbParser.save_multimer(prot_data, pdb_path, pred_info=pred_info)

        # export the predicted sequence to a FASTA file
        if any(['X' in x for x in aa_seq_dict.values()]):
            new_aa_seq_dict = self._build_seq_data(inputs, outputs)
            new_fas_fpath = pdb_path.replace('.pdb', '.fasta')
            export_fas_file_mult(new_aa_seq_dict, new_fas_fpath)

        logging.info('[%s] PDB file generated in %.2f second', prot_id, time.time() - start)

    @torch.no_grad()
    def run_batch(self,
                  pid_path,
                  fasta_dir,
                  msa_path,
                  pdb_dir,
                  icf_dpath=None,
                  chunk_size=None):
        """Run the antibody & antigen structure predictor in batches.

        Args:
            pid_fpath: path to the plain-text file of antibody IDs
            fas_dpath: directory path to input antibody FASTA files
            pdb_dpath: directory path to output PDB files
            icf_dpath: directory path to inter-chain feature files
        """
        with open(msa_path) as f:
            msa, _ = parse_a3m(f.read())

        antigen_name = os.path.basename(msa_path).split('.')[0]
        logging.info(f'{antigen_name} antigen feature predicted using pretrained structure prediction model')
        # get antigen feature w/ af2
        ag_feat, _ = self.psp_featurizer(msa_path)

        with open(pid_path, 'r', encoding='UTF-8') as i_file:
            prot_ids = [i_line.strip() for i_line in i_file]

        for prot_id in prot_ids:
            start = time.time()
            fas_fpath = os.path.join(fasta_dir, f'{prot_id}.fasta')
            pdb_fpath = os.path.join(pdb_dir, f'{prot_id}.pdb')

            aa_seq_dict = parse_fas_file_mult(fas_fpath)
            aa_seq_dict = OrderedDict({k[-1]: v for k, v in aa_seq_dict.items()})
            if 'A' not in aa_seq_dict:  # fill antigen sequence
                aa_seq_dict['A'] = msa[0]

            for chn_id in aa_seq_dict:
                resd_set = set(pc.RESD_NAMES_1C) if chn_id == 'A' else set(pc.RESD_WITH_X)
                if len(set(aa_seq_dict[chn_id]) - resd_set) > 0:
                    logging.warning('[%s] non-standard residue(s) detected; skipping ...', prot_id)
                    return

            # build an input dict
            icf_fpath = None if icf_dpath is None else os.path.join(icf_dpath, f'{prot_id}.pt')
            inputs = self._build_inputs(prot_id, aa_seq_dict, icf_fpath)
            inputs = to_device(inputs, self.device)

            # get initial single & pair features
            ligand_id, _ = get_complex_id(inputs)
            aa_seq_dict_ligand = {x: inputs[x]['base']['seq'] for x in ligand_id.split('-')}

            if 'mask' not in inputs[ligand_id]['base']:
                plm_outputs = self.plm_featurizer(aa_seq_dict_ligand)
            else:
                mask_vecs = [inputs[x]['base']['mask'] for x in ligand_id.split('-')]
                plm_outputs = self.plm_featurizer(aa_seq_dict_ligand, mask_prob=1, mask_vecs=mask_vecs)

            inputs[ligand_id]['feat']['sfea'] = plm_outputs['sfea']
            inputs[ligand_id]['feat']['pfea'] = plm_outputs['pfea']
            inputs['A']['feat'] = ag_feat
            outputs = self.model(inputs, chunk_size=chunk_size)
            # export the predicted structure to a PDB file
            prot_data, pred_info = self._build_prot_data(inputs, outputs)
            PdbParser.save_multimer(prot_data, pdb_fpath, pred_info=pred_info)

            # export the predicted sequence to a FASTA file
            if any(['X' in x for x in aa_seq_dict.values()]):
                new_aa_seq_dict = self._build_seq_data(inputs, outputs)
                new_fas_fpath = pdb_fpath.replace('.pdb', '.fasta')
                export_fas_file_mult(new_aa_seq_dict, new_fas_fpath)

            logging.info('[%s] PDB file generated in %.2f second', prot_id, time.time() - start)

    @classmethod
    def _build_inputs(cls, prot_id, aa_seq_dict, icf_fpath=None):
        """Build an input dict for model inference."""
        # obtain chain IDs
        num_chains = len(aa_seq_dict)
        chain_ids = sorted(list(aa_seq_dict.keys()))
        assert all(x in {'H', 'L', 'A'} for x in chain_ids), 'chain ID must be "H", "L" or "A"'
        assert num_chains in (2, 3), f'FASTA file should contain 2 or3 chains: {aa_seq_dict}'
        # build an input dict
        inputs = {'base': {'id': prot_id, 'chain_ids': chain_ids}}
        if num_chains == 2:  # nanobody
            ligand_id = 'H'
            inputs[ligand_id] = {'base': {'seq': aa_seq_dict[ligand_id]}, 'feat': {}}
        else:  # antibody
            for chn_id in ['H', 'L']:
                inputs[chn_id] = {'base': {'seq': aa_seq_dict[chn_id]}, 'feat': {}}
            ligand_id = '-'.join(['H', 'L'])  # pseudo chain ID for the VH-VL complex
            inputs[ligand_id] = {
                'base': {'seq': aa_seq_dict['H'] + aa_seq_dict['L']},
                'asym_id': get_asym_ids([aa_seq_dict['H'], aa_seq_dict['L']]).unsqueeze(dim=0),
                'feat': {},
            }
        # prepare for sequence recovery
        if 'X' in inputs[ligand_id]['base']['seq']:
            mask_vec = torch.LongTensor([resd == 'X' for resd in inputs[ligand_id]['base']['seq']])
            inputs[ligand_id]['base']['mask'] = get_mlm_masks(
                [inputs[ligand_id]['base']['seq']], mask_prob=1, mask_vecs=[mask_vec])
            if num_chains == 3:
                inputs['H']['base']['mask'] = inputs[ligand_id]['base']['mask'][:len(inputs['H']['base']['seq'])]
                inputs['L']['base']['mask'] = inputs[ligand_id]['base']['mask'][len(inputs['H']['base']['seq']):]

        # antigen
        inputs['A'] = {'base': {'seq': aa_seq_dict['A']}, 'feat': {}}
        complex_id = ':'.join([ligand_id, 'A'])
        aa_seq_ligand = inputs[ligand_id]['base']['seq']
        aa_seq_receptor = inputs['A']['base']['seq']

        inputs[complex_id] = {
            'base': {'seq': aa_seq_ligand + aa_seq_receptor},
            'asym_id': get_asym_ids([aa_seq_ligand, aa_seq_receptor]).unsqueeze(dim=0),
            'feat': {},
        }

        if icf_fpath is not None:
            ic_feat = torch.load(icf_fpath)
            ic_feat = ic_feat.unsqueeze(dim=0).unsqueeze(dim=-1)
            inputs[complex_id]['feat']['icf'] = ic_feat.to(torch.float)

        return inputs

    @classmethod
    def _build_prot_data(cls, inputs, outputs):
        """Build a dict of protein structure data."""
        pred_info = 'REMARK 250 Structure predicted by tFold-Ag\n'
        ligand_id, receptor_id = get_complex_id(inputs)
        device = inputs[ligand_id]['feat']['sfea'].device
        ligand_ids = ligand_id.split('-')
        chn_ids = ligand_ids + [receptor_id]
        complex_id = ':'.join([ligand_id, receptor_id])
        prot_data = OrderedDict()
        start = 0

        # build a dict of protein structure data
        for chn_id in chn_ids:
            aa_seq = inputs[chn_id]['base']['seq']
            if 'X' in aa_seq:  # for sequence recovery
                aa_seq = outputs[complex_id]['1d']['seq'][start:start + len(aa_seq)]
                pred_info += f'REMARK 250 Predicted Sequence for chain {chn_id}: {aa_seq}\n'
            prot_data[chn_id] = {
                'seq': aa_seq,
                'cord': outputs[complex_id]['3d']['cord'][-1][start:start + len(aa_seq)].detach(),
                'cmsk': ProtStruct.get_cmsk_vld(aa_seq, device),
                'plddt': outputs[complex_id]['3d']['plddt'][-1]['plddt-r'][start:start + len(aa_seq)].detach(),
            }
            start += len(aa_seq)

        iptm_score = outputs[complex_id]['3d']['tmsc_dict']['iptm'].item()
        ptm_score = outputs[complex_id]['3d']['tmsc_dict']['ptm'].item()
        pred_info += f'REMARK 250 Predicted ipTM score: {iptm_score:.4f}\n'
        pred_info += f'REMARK 250 Predicted pTM score: {ptm_score:.4f}\n'
        plddt_receptor = torch.mean(prot_data[receptor_id]['plddt']).item()
        pred_info += f'REMARK 250 Antigen Predicted lDDT-Ca score: {plddt_receptor:.4f}\n'

        return prot_data, pred_info

    @classmethod
    def _build_seq_data(cls, inputs, outputs):
        """Build a dict of protein sequence data."""
        ligand_id, receptor_id = get_complex_id(inputs)
        complex_id = ':'.join([ligand_id, receptor_id])
        ligand_ids = ligand_id.split('-')
        start = 0
        aa_seq_dict = OrderedDict()

        # build a dict of protein sequence data
        for chn_id in ligand_ids:
            aa_seq = inputs[chn_id]['base']['seq']
            if 'X' in aa_seq:  # for sequence recovery
                aa_seq = outputs[complex_id]['1d']['seq'][start:start + len(aa_seq)]
                plddt = outputs[complex_id]['3d']['plddt'][-1]['plddt-r'][start:start + len(aa_seq)].detach()
                mask = inputs[ligand_id]['base']['mask'][start:start + len(aa_seq)]
                region_plddt = torch.sum(plddt * mask) / (torch.sum(mask) + 1e-6)
                key = '%s | design confidence %.3f' % (chn_id, region_plddt)
            else:
                key = chn_id
            aa_seq_dict[key] = aa_seq
            start += len(aa_seq)

        return aa_seq_dict

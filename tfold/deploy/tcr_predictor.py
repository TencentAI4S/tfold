# Copyright (c) 2024, Tencent Inc. All rights reserved.
# Data: 2024/6/7 14:45
import logging
from collections import OrderedDict

import torch

from tfold.model.arch import PPIModel, TCRpMHCModel
from tfold.protein import ProtStruct, export_fasta
from tfold.protein.data_transform import get_asym_ids
from tfold.protein.parser import PdbParser
from tfold.protein.utils import get_mlm_masks
from tfold.utils import to_device
from .base_structure_predictor import BaseStructurePredictor
from .plm_complex_predictor import PLMComplexPredictor


class PeptideMHCPredictor(PLMComplexPredictor):
    """The pMHC multimer structure predictor."""

    def infer(self, chains, *args, **kwargs):
        assert len(chains) in (1, 2, 3), f"expect 2 or 3 chains, but got {len(chains)}"
        return super().infer(chains, *args, **kwargs)


class TCRPredictor(PLMComplexPredictor):
    """The TCR structure predictor."""

    def infer(self, chains, *args, **kwargs):
        assert len(chains) in (1, 2), f"expect 1 or 2 chains, but got {len(chains)}"
        return super().infer(chains, *args, **kwargs)


class TCRpMHCPredictor(BaseStructurePredictor):
    """The TCR-pMHC multimer structure predictor."""

    def __init__(self, ppi_path, trunk_path):
        super(TCRpMHCPredictor, self).__init__()
        logging.info('restoring the pre-trained tFold-PPI-SeqPT model ...')
        self.plm_featurizer = PPIModel.restore(ppi_path)
        logging.info('restoring the pre-trained tFold-TCR model ...')
        self.model = TCRpMHCModel.restore(trunk_path)
        self.eval()

    def forward(self, inputs, chunk_size=None):
        inputs = to_device(inputs, self.device)
        # TCR: ligand
        ligand_id = inputs["base"]["ligand_id"]
        ligand_seqs = [inputs[x]['base']['seq'] for x in ligand_id.split(':')]
        plm_outputs = self.plm_featurizer(ligand_seqs)
        inputs[ligand_id]['feat']['sfea'] = plm_outputs['sfea']
        inputs[ligand_id]['feat']['pfea'] = plm_outputs['pfea']
        # pMHC:
        receptor_id = inputs["base"]["receptor_id"]
        receptor_seqs = [inputs[x]['base']['seq'] for x in receptor_id.split(':')]
        plm_outputs = self.plm_featurizer(receptor_seqs)
        inputs[receptor_id]['feat']['sfea'] = plm_outputs['sfea']
        inputs[receptor_id]['feat']['pfea'] = plm_outputs['pfea']
        outputs = self.model(inputs, chunk_size=chunk_size)

        return outputs

    def merge_chains(self, inputs, chain_ids):
        sequences = [inputs[chain_id]["base"]["seq"] for chain_id in chain_ids]
        data = {
            "base": {"seq": ''.join(sequences)},
            "asym_id": get_asym_ids(sequences).unsqueeze(dim=0),
            "feat": {}
        }

        return data

    def _build_inputs(self, chains):
        """Build an input dict for model."""
        inputs = super()._build_inputs(chains)
        # TCR
        if "B" in inputs and "A" in inputs:
            inputs["B:A"] = self.merge_chains(inputs, ["B", "A"])
            inputs["base"]["ligand_id"] = "B:A"

        if "D" in inputs and "G" in inputs:
            inputs["D:G"] = self.merge_chains(inputs, ["D", "G"])
            inputs["base"]["ligand_id"] = "D:G"

        # prepare receptor inputs for pMHC
        if 'N' in inputs:
            inputs["M:N"] = self.merge_chains(inputs, ["M", "N"])
            inputs["base"]["mhc_id"] = "M:N"
        else:
            inputs["base"]["mhc_id"] = "M"

        # pMHC
        mhc_id = inputs["base"]["mhc_id"]
        if "P" in inputs:
            chain_ids = list(mhc_id.split(":")) + ["P", ]
            receptor_id = ":".join(chain_ids)
            inputs["base"]["peptide_id"] = "P"
            inputs[receptor_id] = self.merge_chains(inputs, chain_ids)
        else:
            receptor_id = mhc_id

        inputs["base"]["receptor_id"] = receptor_id

        ligand_id = inputs["base"]["ligand_id"]

        if 'X' in inputs[ligand_id]['base']['seq']:
            mask_vec = torch.LongTensor([resd == 'X' for resd in inputs[ligand_id]['base']['seq']])
            inputs[ligand_id]['base']['mask'] = get_mlm_masks(
                [inputs[ligand_id]['base']['seq']], mask_prob=1, mask_vecs=[mask_vec])

            if len(ligand_id.split(':')) == 2:
                inputs['B']['base']['mask'] = inputs[ligand_id]['base']['mask'][:len(inputs['B']['base']['seq'])]
                inputs['A']['base']['mask'] = inputs[ligand_id]['base']['mask'][len(inputs['B']['base']['seq']):]

        chain_ids = ligand_id.split(":") + receptor_id.split(":")
        complex_id = ":".join(chain_ids)
        inputs[complex_id] = self.merge_chains(inputs, chain_ids)
        inputs["base"]["complex_id"] = complex_id

        return inputs

    def output_to_pdb(self, inputs, outputs, filename):
        """Build a dict of protein structure data."""
        pred_info = 'REMARK 250 Structure predicted by tFold\n'
        complex_id = inputs["base"]["complex_id"]

        prot_data = OrderedDict()
        start = 0
        for chn_id in complex_id.split(":"):
            aa_seq = inputs[chn_id]['base']['seq']
            if 'X' in aa_seq:  # for sequence recovery
                aa_seq = outputs[complex_id]['1d']['seq'][start:start + len(aa_seq)]
                pred_info += f'REMARK 250 Predicted Sequence for chain {chn_id}: {aa_seq}\n'

            prot_data[chn_id] = {
                'seq': aa_seq,
                'cord': outputs[complex_id]['3d']['cord'][-1][start:start + len(aa_seq)],
                'cmsk': ProtStruct.get_cmsk_vld(aa_seq, self.device),
                'plddt': outputs[complex_id]['3d']['plddt'][-1]['plddt-r'][start:start + len(aa_seq)],
            }
            start += len(aa_seq)

        iptm_score = outputs[complex_id]['3d']['tmsc_dict']['iptm'].item()
        ptm_score = outputs[complex_id]['3d']['tmsc_dict']['ptm'].item()
        pred_info += f'REMARK 250 Predicted ipTM score: {iptm_score:.4f}\n'
        pred_info += f'REMARK 250 Predicted pTM score: {ptm_score:.4f}\n'

        if 'actifptm' in outputs[complex_id]['3d']['tmsc_dict']:
            actifptm_score = outputs[complex_id]['3d']['tmsc_dict']['actifptm'].item()
            pred_info += f'REMARK 250 Predicted actifptm score: {actifptm_score:.4f}\n'

        PdbParser.save_multimer(prot_data, filename, pred_info=pred_info)
        logging.info(f'PDB file generated: {filename}')

    def output_to_fasta(self, inputs, outputs, filename):
        """export the predicted sequence to a FASTA file"""
        complex_id = inputs["base"]["complex_id"]
        ligand_id = inputs["base"]["ligand_id"]
        ligand_ids = ligand_id.split(':')
        start = 0
        sequences = []
        ids = []
        descriptions = []
        for chn_id in ligand_ids:
            aa_seq = inputs[chn_id]['base']['seq']
            if 'X' in aa_seq:  # for sequence recovery
                aa_seq = outputs[complex_id]['1d']['seq'][start:start + len(aa_seq)]
                plddt = outputs[complex_id]['3d']['plddt'][-1]['plddt-r'][start:start + len(aa_seq)].detach()
                mask = inputs[ligand_id]['base']['mask'][start:start + len(aa_seq)]
                region_plddt = torch.sum(plddt * mask) / (torch.sum(mask) + 1e-6)
                desc = f"design confidence {region_plddt:.3f}"
            else:
                desc = ""

            descriptions.append(desc)
            sequences.append(aa_seq)
            ids.append(chn_id)
            start += len(aa_seq)

        export_fasta(sequences, ids=ids, descriptions=descriptions, output=filename)

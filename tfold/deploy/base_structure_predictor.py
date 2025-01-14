# Copyright (c) 2024, Tencent Inc. All rights reserved.
# Data: 2024/6/7 14:45
import logging
from collections import OrderedDict

import torch

from tfold.model import BaseModel
from tfold.protein import prot_constants as rc, ProtStruct
from tfold.protein.data_transform import get_asym_ids
from tfold.protein.parser import PdbParser


class BaseStructurePredictor(BaseModel):

    def _build_inputs(self, chains):
        """Build an input dict for model inference."""
        assert len(chains) > 0
        chain_ids = [chain["id"] for chain in chains]
        inputs = {"base": {"chain_ids": chain_ids}}
        for chain in chains:
            chain_id = chain["id"]
            sequence = chain["sequence"]
            # valid aa sequence check
            if 'X' not in sequence:
                assert set(sequence).issubset(set(rc.RESD_NAMES_1C)), f"{chain_id} not in standard aa sequence"
            inputs[chain_id] = {"base": {"seq": sequence}, "feat": {}}
            if "msa" in chain:
                inputs[chain_id]["base"]["msa"] = chain["msa"]
                inputs[chain_id]["base"]["deletion_matrix"] = chain["deletion_matrix"]

        # multimer information
        if len(chains) > 1:
            complex_id = ":".join(chain_ids)
            inputs["base"]["complex_id"] = complex_id
            sequences = [chain["sequence"] for chain in chains]
            inputs[complex_id] = {
                "base": {"seq": ''.join(sequences)},
                "asym_id": get_asym_ids(sequences).unsqueeze(dim=0),
                "feat": {}
            }

        return inputs

    def _split_complex_outputs(self, inputs, complex_outputs, chain_ids=None):
        """Split the dict of output tensors into primary & secondary chains.

        Notes:
            Scalar outputs are not re-calculated for primary & secondary chains for simplicity.
        """
        complex_id = inputs["base"]["complex_id"]
        if chain_ids is None:
            chain_ids = complex_id.split(":")

        keys_2d = ['cb', 'om', 'th', 'ph']
        keys_3d = ['quat', 'trsl', 'angl', 'quat-u']
        start = 0
        outputs = {}
        for chain_id in chain_ids:
            seq = inputs[chain_id]['base']['seq']
            seq_len = len(seq)
            out_2d = {
                k: complex_outputs['2d'][k][:, :, start:start + seq_len, start:start + seq_len] for k in keys_2d}

            params_3d = [{k: x[k][start:start + seq_len] for k in keys_3d} for x in complex_outputs['3d']['params']]
            plddt = [{k: x[k][start:start + seq_len] for k in ('logit', 'plddt-r')} for x in
                     complex_outputs['3d']['plddt']]
            cord = [x[start:start + seq_len] for x in complex_outputs['3d']['cord']]
            tmsc_dict = {'ptm_logt': complex_outputs['3d']['tmsc_dict']['ptm_logt'][start:start + seq_len,
                                                                                    start:start + seq_len]}
            outputs[chain_id] = {
                "seq": seq,
                "sfea": complex_outputs['sfea'][:, start:start + seq_len],
                "pfea": complex_outputs['pfea'][:, start:start + seq_len, start:start + seq_len],
                "cord": complex_outputs['cord'][start:start + seq_len],
                '2d': out_2d,
                '3d': {
                    "params": params_3d,
                    "plddt": plddt,
                    "cord": cord,
                    "tmsc_dict": tmsc_dict,
                    "fram_sc": complex_outputs['3d']['fram_sc'][start:start + seq_len]
                },
            }
            start += seq_len

        outputs[complex_id] = complex_outputs
        return outputs

    @torch.no_grad()
    def infer(self, chains, *args, **kwargs):
        """inferece monomer chain or multimer chains"""
        inputs = self._build_inputs(chains)
        outputs = self.forward(inputs, *args, **kwargs)
        return inputs, outputs

    def output_to_pdb(self, inputs, outputs, filename):
        """save the pdb from the model given the model output."""
        pred_info = 'REMARK 250 Structure predicted by tFold\n'
        complex_id = inputs["base"].get("complex_id", inputs["base"]["chain_ids"][0])
        chain_ids = complex_id.split(":")

        chains = OrderedDict()
        for chain_id in chain_ids:
            sequence = inputs[chain_id]["base"]["seq"]
            chains[chain_id] = {
                "seq": sequence,
                "cord": outputs[chain_id]['cord'],
                "cmsk": ProtStruct.get_cmsk_vld(sequence, self.device),
                "plddt": outputs[chain_id]['3d']['plddt'][-1]["plddt-r"]
            }
            if "ptm" in outputs[chain_id]["3d"]["tmsc_dict"]:
                ptm = outputs[chain_id]["3d"]["tmsc_dict"]["ptm"]
                chains[chain_id]["ptm"] = ptm
                pred_info += f"REMARK 250 Predicted pTM score: {ptm.item():.4f}\n"

        if len(chain_ids) > 1:
            complex_id = ":".join(chain_ids)
            if "ptm" in outputs[complex_id]["3d"]["tmsc_dict"]:
                ptm = outputs[complex_id]["3d"]["tmsc_dict"]["ptm"]
                pred_info += f"REMARK 250 Predicted pTM score: {ptm.item():.4f}\n"

            if "iptm" in outputs[complex_id]["3d"]["tmsc_dict"]:
                iptm = outputs[complex_id]["3d"]["tmsc_dict"]["iptm"]
                pred_info += f"REMARK 250 Predicted ipTM score: {iptm.item():.4f}\n"

        PdbParser.save_multimer(chains, filename, pred_info=pred_info)
        logging.info(f'PDB file generated: {filename}')

    def infer_pdb(self, chains, filename, *args, **kwargs):
        inputs, outputs = self.infer(chains, *args, **kwargs)
        self.output_to_pdb(inputs, outputs, filename)

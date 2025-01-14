# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
import logging

import torch

from tfold.config import get_config, CfgNode
from ..base_model import BaseModel
from ..core import ComplexStructureModel, DockingModelSM
from ...build import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class AgModel(BaseModel):
    """The receptor & ligand complex prediction model.
    """

    def __init__(self, config):
        super().__init__()
        ligand_cfg = config.ligand
        docking_cfg = config.docking
        self.ligand_model = ComplexStructureModel(
            **ligand_cfg.to_dict()
        )
        self.docking_model = DockingModelSM(
            **docking_cfg.to_dict()
        )

    @classmethod
    def restore(cls, path):
        """Restore a pre-trained model."""
        state = torch.load(path, map_location='cpu')
        config = get_config()
        config.update(CfgNode(state['config']))
        logging.info(config)
        model = cls(config.model)
        model.load_state_dict(state['model'])
        logging.info('restore the pre-trained tFold-Ag model %s', path)
        return model

    def forward(self, inputs, chunk_size=None):
        """

        Args:
            inputs: dict of input tensors

        Returns:
            outputs: dict of model predictions

        Notes:
        * The input dict is organized as below (unused data entries are omitted here):
          > ligand-chn1:
            > base:
              > seq: ligand first chain's amino-acid sequence of length L1
            > feat:
              > sfea: ligand first chain's initial single features of size 1 x L1 x D_s
              > pfea: ligand first chain's initial pair features of size 1 x L1 x L1 x D_p
          > ligand-chn2: (optional; only available for multimer ligand)
            > base:
              > seq: ligand second chain's amino-acid sequence of length L2
          > ligand chn1-chn2: (optional; only available for multimer ligand)
            > base:
              > seq: ligand's amino-acid sequence of length Ll = (L1 + L2) (no linker)
            > asym_id: asymmetric ID of length Ll
            > feat: (only used for multimer inputs)
              > sfea: ligand second chain's initial single features of size 1 x Ll x D_s
              > pfea: ligand second chain's initial pair features of size 1 x Ll x Ll x D_p
          > receptor: (optional; only available for ligand-receptor complex)
            > base:
              > seq: receptor chain's amino-acid sequence of length Lr
            > feat:
              > mfea: receptor chain's initial MSA features of size 1 x M x Lr x D_s
              > pfea: receptor chain's initial pair features of size 1 x Lr x Lr x D_p
              > cord: receptor chain's initial coordinate of size 1 x Lr x 14
         > complex ligand:receptor: (optional; only available if receptor exists)
          > base:
            > seq: complex's amino-acid sequence of length Lc = (Ll + Lr) (no linker)
          > asym_id: asymmetric ID of length Lc
        """
        ligand_id = inputs["base"]["ligand_id"]
        receptor_id = inputs["base"].get("receptor_id", None)

        sequences = []
        for chn_id in ligand_id.split(':'):
            seq = inputs[chn_id]['base']['seq']
            if 'mask' in inputs[chn_id]['base']:
                mask = inputs[chn_id]['base']['mask']
                modified_seq = ''.join(['G' if mask[i] == 1 else seq[i] for i in range(len(seq))])
                seq = modified_seq
            sequences.append(seq)

        s = inputs[ligand_id]["feat"]["sfea"]
        z = inputs[ligand_id]["feat"]["pfea"]

        outputs = {}
        with torch.no_grad():
            outputs[ligand_id] = self.ligand_model(sequences, s, z, chunk_size=chunk_size)  # ligand_id

        if receptor_id is not None:  # ligand & receptor complex
            inputs_dock = {"base": {}}
            inputs_dock["base"]["receptor_id"] = receptor_id
            inputs_dock["base"]["ligand_id"] = ligand_id
            inputs_dock[receptor_id] = inputs[receptor_id]
            # prepare the ligand input for docking module
            inputs_dock[ligand_id] = {
                'base': inputs[ligand_id]['base'],
                'feat': {
                    'sfea': outputs[ligand_id]['sfea'],
                    'pfea': outputs[ligand_id]['pfea'],
                    'cord': outputs[ligand_id]['cord'].detach()
                }
            }
            complex_id = ':'.join([ligand_id, receptor_id])
            inputs_dock[complex_id] = inputs[complex_id]
            outputs_cp = self.docking_model(inputs_dock, chunk_size=chunk_size)
            outputs[complex_id] = outputs_cp

        return outputs
# Copyright (c) 2024, Tencent Inc. All rights reserved.
# Data: 2024/6/5 17:15

import torch

from tfold.config import get_config, CfgNode
from ..base_model import BaseModel
from ..core import ComplexStructureModel, DockingModelSS as DockingModel


class TCRpMHCModel(BaseModel):
    """TCR prediction model.
    """
    def __init__(self, cfg):
        super().__init__()
        # tFold-Ab model
        ligand_cfg = CfgNode()
        ligand_cfg.model = cfg.ligand
        self.ligand_model = ComplexStructureModel(ligand_cfg)
        # tFold-pMHC model
        receptor_cfg = CfgNode()
        receptor_cfg.model = cfg.receptor
        self.receptor_model = ComplexStructureModel(receptor_cfg)
        self.docking_model = DockingModel(**cfg.docking.to_dict())

    @classmethod
    def restore(cls, path):
        state = torch.load(path, map_location='cpu')
        config = get_config()
        config.update(state['config'])
        print(config)
        model = cls(config.model)
        model.load_state_dict(state['model'])
        return model

    def forward(self, inputs, num_recycles=None, chunk_size=None):
        """
        Args:
            inputs: dict of input tensors

        Returns:
            outputs: dict of model predictions

        Notes:
        * The input dict is organized as below (unused data entries are omitted here):
          > [chain_id]:
            > base:
              > seq: ligand first chain's amino-acid sequence of length L1
            > feat:
              > sfea: ligand first chain's initial single features of size 1 x L1 x D_s
              > pfea: ligand first chain's initial pair features of size 1 x L1 x L1 x D_p
          > receptor: (optional; only available for ligand-receptor complex)
            > base:
              > seq: receptor chain's amino-acid sequence of length Lr
            > feat:
              > sfea: receptor chain's initial MSA features of size 1 x Lr x D_s
              > pfea: receptor chain's initial pair features of size 1 x Lr x Lr x D_p
              > cord: receptor chain's initial coordinate of size 1 x Lr x 14
         > complex ligand:receptor: (optional; only available if receptor exists)
          > base:
            > seq: complex's amino-acid sequence of length Lc = (Ll + Lr) (no linker)
          > asym_id: asymmetric ID of length Lc
        """
        # run the pretrained model to compute TCR & pMHC feature
        with torch.set_grad_enabled(False):
            ligand_id = inputs["base"]["ligand_id"]
            ligand_sequences = []
            for chn_id in ligand_id.split(':'):
                seq = inputs[chn_id]['base']['seq']
                if 'mask' in inputs[chn_id]['base']:
                    mask = inputs[chn_id]['base']['mask']
                    modified_seq = ''.join(['G' if mask[i] == 1 else seq[i] for i in range(len(seq))])
                    seq = modified_seq
                ligand_sequences.append(seq)

            # TCR
            li_outputs = self.ligand_model(ligand_sequences,
                                           s_init=inputs[ligand_id]["feat"]["sfea"],
                                           z_init=inputs[ligand_id]["feat"]["pfea"],
                                           asym_id=inputs[ligand_id].get("asym_id", None).squeeze(0),
                                           num_recycles=num_recycles,
                                           chunk_size=chunk_size)

            # pMHC
            receptor_id = inputs["base"]["receptor_id"]
            receptor_sequences = []
            for chn_id in receptor_id.split(':'):
                seq = inputs[chn_id]['base']['seq']
                if 'mask' in inputs[chn_id]['base']:
                    mask = inputs[chn_id]['base']['mask']
                    modified_seq = ''.join(['G' if mask[i] == 1 else seq[i] for i in range(len(seq))])
                    seq = modified_seq
                receptor_sequences.append(seq)
            re_outputs = self.receptor_model(receptor_sequences,
                                             s_init=inputs[receptor_id]["feat"]["sfea"],
                                             z_init=inputs[receptor_id]["feat"]["pfea"],
                                             asym_id=inputs[receptor_id].get("asym_id", None).squeeze(0),
                                             num_recycles=num_recycles,
                                             chunk_size=chunk_size)

        # complex for TCR-pMHC
        complex_id = inputs["base"]["complex_id"]

        inputs_dock = {"base": {"ligand_id": ligand_id, "receptor_id": receptor_id}}
        inputs_dock[receptor_id] = inputs[receptor_id]
        inputs_dock[ligand_id] = inputs[ligand_id]
        inputs_dock[complex_id] = inputs[complex_id]

        # prepare the ligand input for docking module
        inputs_dock[ligand_id]['feat'] = {
            'sfea': li_outputs['sfea'],
            'pfea': li_outputs['pfea'],
            'cord': li_outputs['cord'].detach()
        }

        inputs_dock[receptor_id]['feat'] = {
            'sfea': re_outputs['sfea'],
            'pfea': re_outputs['pfea'],
            'cord': re_outputs['3d']['cord'][-1].detach(),
        }

        outputs = {ligand_id: li_outputs}
        outputs[complex_id] = self.docking_model(inputs_dock, num_recycles=num_recycles, chunk_size=chunk_size)

        return outputs

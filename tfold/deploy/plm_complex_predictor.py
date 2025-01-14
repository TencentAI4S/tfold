# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
import logging

import torch

from tfold.config import get_config, CN
from tfold.model import ComplexLiteModel
from tfold.model.pretrain import load_model_hub
from tfold.utils import to_device
from .base_structure_predictor import BaseStructurePredictor


class PLMComplexPredictor(BaseStructurePredictor):
    """plm based complex structure predictor."""

    def __init__(self, cfg):
        super(PLMComplexPredictor, self).__init__(cfg)
        self.model = ComplexLiteModel(cfg)
        self.eval()

    @classmethod
    def restore(cls, path):
        state = torch.load(path, map_location='cpu')
        config = get_config()
        config.update(state['config'])
        logging.info(config)
        model = cls(config)
        model.model.load_state_dict(state['model'])
        return model

    @classmethod
    def restore_from_hub(cls, name):
        return cls.restore(load_model_hub(name))

    @classmethod
    def restore_from_module(cls, ppi_path=None, trunk_path=None):
        logging.info('restoring the pre-trained ppi model ...')
        ppi_state = torch.load(ppi_path, map_location='cpu')
        cfg = get_config()
        cfg.ppi = CN(ppi_state['config'])

        logging.info('restoring the pre-trained tfold ab model ...')
        trunk_state = torch.load(trunk_path, map_location='cpu')
        cfg.trunk = CN(trunk_state['config'])

        model = cls(cfg)
        model.model.ppi.load_state_dict(ppi_state['model'])
        model.model.trunk.load_state_dict(trunk_state['model'])
        return model

    def forward(self, inputs, *args, **kwargs):
        """Run the antibody structure predictor.
        """
        # default complex id is first chain
        complex_id = inputs["base"].get("complex_id", inputs["base"]['chain_ids'][0])

        chain_ids = complex_id.split(":")
        inputs = to_device(inputs, self.device)
        sequences = [inputs[cid]["base"]["seq"] for cid in chain_ids]
        is_mono = len(sequences) == 1
        asym_id = None if is_mono else inputs[complex_id]['asym_id'][0]
        complex_outputs = self.model(sequences, asym_id=asym_id, *args, **kwargs)

        if is_mono:
            outputs = {
                complex_id: complex_outputs
            }
        else:
            outputs = self._split_complex_outputs(inputs, complex_outputs)

        return outputs

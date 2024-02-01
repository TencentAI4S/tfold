# -*- coding: utf-8 -*-
# Copyright (c) 2023, Tencent Inc. All rights reserved.
import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn
from openfold.data import data_pipeline, feature_pipeline
from openfold.model.model import AlphaFold

from tfold.config import CfgNode
from tfold.protein.atom_mapper import AtomMapper
from tfold.protein.parser import parse_a3m
from tfold.utils import all_logging_disabled
from tfold.utils.tensor import tensor_tree_map


def template_feats_placeholder():
    return {
        'template_aatype': np.array([], np.int64),
        'template_all_atom_masks': np.zeros([], np.float32),
        'template_all_atom_positions': np.zeros([], np.float32),
        'template_domain_names': np.zeros([], np.object),
    }


class PspFeaturizer(nn.Module):
    """The pre-trained protein structure prediction (PSP) feature extractor."""

    def __init__(
            self,
            config
    ):
        super().__init__()
        self.config = config
        self.atom_mapper = AtomMapper()
        self.model = AlphaFold(self.config)
        self.model.eval()
        self.num_recycles = self.config.data.common.max_recycling_iters

    @property
    def device(self):
        return next(self.parameters()).device

    @classmethod
    def restore(cls, path):
        state = torch.load(path, map_location='cpu')
        config = CfgNode(state['config'])
        model = cls(config).eval()
        model.model.load_state_dict(state['model'])
        logging.info('model weights restored from %s', path)
        return model

    def _get_feature_dict(self, msa_path, idx_resd_beg=None, idx_resd_end=None):
        if os.path.exists(msa_path):
            with open(msa_path) as f:
                MSA, deletion_matrix = parse_a3m(f.read())
        else:
            raise ValueError(f'<msa_path> {msa_path} is not existed')

        feature_dict = {}
        query_seq = MSA[0]
        if idx_resd_beg is not None and idx_resd_end is not None:
            query_seq = query_seq[idx_resd_beg:idx_resd_end]
            MSA = [s[idx_resd_beg:idx_resd_end] for s in MSA]
            deletion_matrix = [d[idx_resd_beg:idx_resd_end] for d in deletion_matrix]

        feature_dict.update(data_pipeline.make_sequence_features(query_seq, 'test', len(query_seq)))
        feature_dict.update(data_pipeline.make_msa_features([MSA], [deletion_matrix]))
        feature_dict.update(template_feats_placeholder())

        return feature_dict

    def forward(self,
                msa_path,
                idx_resd_beg=None,
                idx_resd_end=None,
                num_recycles=None):
        num_recycles = self.num_recycles if num_recycles is None else num_recycles
        feature_dict = self._get_feature_dict(msa_path, idx_resd_beg, idx_resd_end)
        aa_seq = bytes.decode(feature_dict['sequence'][0])
        feature_processor = feature_pipeline.FeaturePipeline(self.config.data)
        try:
            process_feature_dict = feature_processor.process_features(feature_dict, mode='predict')
        except IndexError as e:
            logging.error(f'Fail to parse idx_resd_beg: {idx_resd_beg}, idx_resd_end: {idx_resd_end}...')
            raise e
        process_feature_dict = {
            k: torch.as_tensor(v, device=self.device)
            for k, v in process_feature_dict.items()
        }
        if self.config.data.common.max_recycling_iters > 0:
            for item in process_feature_dict:
                num_iters = random.sample(range(num_recycles + 1), num_recycles + 1)
                process_feature_dict[item] = process_feature_dict[item][..., num_iters]

        with torch.no_grad():
            with all_logging_disabled(highest_level=logging.WARNING):
                result, ptm = self._forward_impl(process_feature_dict)

        for i in range(num_recycles + 1):
            result[i][2] = self.atom_mapper.run(aa_seq, result[i][2], frmt_src='n37', frmt_dst='n14-tf')
            result[i][3] = self.atom_mapper.run(aa_seq, result[i][3], frmt_src='n37', frmt_dst='n14-tf')

        result_dict = {
            'mfea': torch.stack([result[i][0] for i in range(num_recycles + 1)], dim=0),
            'pfea': torch.stack([result[i][1] for i in range(num_recycles + 1)], dim=0),
            'cord': torch.stack([result[i][2] for i in range(num_recycles + 1)], dim=0),
            'cmsk': torch.stack([result[i][3] for i in range(num_recycles + 1)], dim=0),
        }

        return result_dict, ptm

    def _forward_impl(self, batch):
        # Initialize recycling embeddings
        m_1_prev, z_prev, x_prev = None, None, None
        prevs = [m_1_prev, z_prev, x_prev]
        result = []
        # Main recycling loop
        num_iters = batch['aatype'].shape[-1]
        for cycle_no in range(num_iters):
            # Select the features for the current recycling cycle
            fetch_cur_batch = lambda t: t[..., cycle_no]  # noqa
            feats = tensor_tree_map(fetch_cur_batch, batch)
            # Enable grad iff we're training and it's the final recycling layer
            is_final_iter = cycle_no == (num_iters - 1)
            with torch.no_grad():
                # Run the next iteration of the model
                outputs, m_1_prev, z_prev, x_prev = self.model.iteration(feats, prevs, _recycle=(num_iters > 1))
                feat_tns = [
                    outputs['msa'].clone().cpu(),
                    outputs['pair'].clone().cpu(),
                    outputs['final_atom_positions'].clone().cpu(),
                    outputs['final_atom_mask'].clone().cpu()
                ]
                result.append(feat_tns)
                if not is_final_iter:
                    del outputs
                    prevs = [m_1_prev, z_prev, x_prev]
                    del m_1_prev, z_prev, x_prev

        outputs.update(self.model.aux_heads(outputs))
        ptm = None
        if 'predicted_tm_score' in outputs:
            ptm = outputs['predicted_tm_score'].item()

        return result, ptm

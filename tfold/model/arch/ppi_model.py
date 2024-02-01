# -*- coding: utf-8 -*-
# Copyright (c) 2023, Tencent Inc. All rights reserved.
import logging

import torch
from torch import nn

from tfold.config import configurable, CfgNode
from tfold.protein.utils import get_asym_ids, get_enty_ids, get_mlm_masks, apply_mlm_masks
from tfold.utils import to_device
from .esm2 import ESM2, Alphabet
from ..build import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class PPIModel(nn.Module):
    """The protein-protein interaction (PPI) model."""

    @classmethod
    def from_config(cls, cfg):

        alphabet = Alphabet.from_architecture(cfg.tokenizer)

        return {
            "alphabet": alphabet
        }

    @configurable
    def __init__(self,
                 encoder_embed_dim=1280,
                 encoder_layers=33,
                 encoder_attention_heads=20,
                 token_dropout=True,
                 use_crp_embeddings=True,
                 alphabet=None):
        super().__init__()
        if alphabet is None:
            self.alphabet = Alphabet.from_architecture('ESM-1b')
        else:
            self.alphabet = alphabet

        self.mask_idx = self.alphabet.mask_idx
        self.tokenizer = self.alphabet.get_batch_converter()
        self.model = ESM2(
            num_layers=encoder_layers,
            embed_dim=encoder_embed_dim,
            attention_heads=encoder_attention_heads,
            alphabet=self.alphabet,
            token_dropout=token_dropout,
            use_crp_embeddings=use_crp_embeddings
        )
        self.c_s = encoder_embed_dim
        self.c_z = encoder_layers * encoder_attention_heads
        self.repr_layers = [self.model.num_layers]

    @classmethod
    def restore(cls, path):
        """Restore the pre-trained protein-protein interaction (PPI) model."""
        state = torch.load(path, map_location='cpu')
        config = CfgNode(state['config'])
        model = cls(config)
        model.load_state_dict(state['model'])
        logging.info('model weights restored from %s', path)
        return model

    def _parse_inputs(self, inputs):
        """Parse raw inputs to prepare ESM2's inputs."""
        pad_size = (1, 1)  # 1 prepended & 1 appended tokens

        # build token indices and asymmetric & entity IDs
        aa_seq_list = []
        asym_ids_list = []
        enty_ids_list = []
        mask_vec_list = []
        for raw_data in inputs:
            aa_seq_list.append(''.join(raw_data['seqs']))
            asym_ids_list.append(nn.functional.pad(raw_data['asym'], pad_size))
            enty_ids_list.append(nn.functional.pad(raw_data['enty'], pad_size))
            mask_vec_list.append(nn.functional.pad(raw_data['mask'], pad_size))

        # pad all the inputs to the same length
        _, _, tokn_mat_orig = self.tokenizer([('', x) for x in aa_seq_list])
        asym_ids = nn.utils.rnn.pad_sequence(asym_ids_list, batch_first=True).to(torch.int8)
        enty_ids = nn.utils.rnn.pad_sequence(enty_ids_list, batch_first=True).to(torch.int8)
        mask_mat = nn.utils.rnn.pad_sequence(mask_vec_list, batch_first=True).to(torch.int8)
        tokn_mat_orig = tokn_mat_orig.to(mask_mat.device)

        # apply random masks
        tokn_mat_pert = apply_mlm_masks(tokn_mat_orig, mask_mat, self.tokenizer.alphabet)

        # pack input tensors into a dict
        input_dict = {
            'tokn-o': tokn_mat_orig,
            'tokn-p': tokn_mat_pert,
            'mask': mask_mat,
            'asym': asym_ids,
            'enty': enty_ids,
        }

        return input_dict

    def _forward_impl(self, inputs):
        """
        Args:
            inputs: list of raw input dicts, each dict contains:
              > ids: list of protein IDs
              > seqs: list of per-chain amino-acid sequences
              > asym: asymmetric IDs
              > enty: entity IDs
              > mask: random masks for MLM training
        Returns:
            outputs: dict of output tensors

        Notes:
        * <asym>, <enty>, and <mask> can be generated w/ utility functions in utils.py.
        """
        # determine whether single & pair features should be extracted
        rtn_sp_feats = (len(inputs) == 1)
        device = next(self.model.parameters()).device
        # parse raw inputs to prepare ESM2's inputs
        input_dict = self._parse_inputs(inputs)
        input_dict = to_device(input_dict, device)

        results = self.model(
            input_dict['tokn-p'],
            asym_ids=input_dict['asym'],
            enty_ids=input_dict['enty'],
            repr_layers=self.repr_layers,
            need_head_weights=rtn_sp_feats,
            return_contacts=False
        )

        # pack all the tensors into a dict
        outputs = {
            'labl': input_dict['tokn-o'],
            'mask': input_dict['mask'],
            'pred': results['logits'],
        }

        # extract single & pair features
        if rtn_sp_feats:
            seq_len = len(''.join(inputs[0]['seqs']))
            sfea_mat = results['representations'][self.model.num_layers]  # 1 x L' x D_s
            pfea_tns = results['attentions'].permute(0, 3, 4, 1, 2)  # 1 x L' x L' x M x H
            outputs['sfea'] = sfea_mat[:, 1:-1].view(1, seq_len, self.c_s)
            outputs['pfea'] = pfea_tns[:, 1:-1, 1:-1].view(1, seq_len, seq_len, self.c_z)

        return outputs

    def forward(self, aa_seq_dict, mask_prob=0.0, mask_vecs=None):
        """Extract PLM embeddings as sinlge & pair features.

        Args:
            aa_seq_dict: ordered dict of (ID, sequence) pairs
            mask_prob: (optional) how likely amino-acid tokens are randomly masked out
            mask_vecs: (optional) list of masking-allowed-or-not indicators, each of length L_i

        Returns:
            outputs: dict of output tensors
        """
        chn_ids = sorted(list(aa_seq_dict.keys()))
        aa_seqs = [aa_seq_dict[chn_id] for chn_id in chn_ids]
        raw_data = {
            'ids': chn_ids,
            'seqs': aa_seqs,
            'asym': get_asym_ids(aa_seqs),
            'enty': get_enty_ids(aa_seqs),
            'mask': get_mlm_masks(aa_seqs, mask_prob=mask_prob, mask_vecs=mask_vecs)
        }
        inputs = [raw_data]
        outputs = self._forward_impl(inputs)

        return outputs

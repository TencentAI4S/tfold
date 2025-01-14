# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
# Data: 2023/12/22 11:14
from functools import partial
from typing import Optional, List

import torch
import torch.nn as nn

from tfold.model.layer import LayerNorm, DropoutRowwise, DropoutColumnwise
from tfold.model.module.attention import (
    GatedMultiheadAttention as Attention,
    TriangleAttentionStartingNode, TriangleAttentionEndingNode,
    TriangleMultiplicationOutgoing, TriangleMultiplicationIncoming
)
from tfold.model.module.evoformer.evoformer_msa import Transition
from tfold.model.utils import checkpoint_blocks, chunk_layer
from tfold.utils.tensor import permute_final_dims


class TemplatePointwiseAttention(nn.Module):
    """
    Implements Algorithm 17.
    """

    def __init__(self,
                 c_t=64,
                 c_z=128,
                 num_heads=4,
                 inf=1e5):
        """
        Args:
            c_t: Template embedding channel dimension
            c_z: Pair embedding channel dimension
        """
        super(TemplatePointwiseAttention, self).__init__()
        self.c_t = c_t
        self.c_z = c_z
        self.num_heads = num_heads
        self.inf = inf
        self.mha = Attention(
            c_q=self.c_z,
            c_k=self.c_t,
            c_v=self.c_t,
            num_heads=self.num_heads,
            gating=False
        )

    def _chunk(self,
               z: torch.Tensor,
               t: torch.Tensor,
               biases: List[torch.Tensor],
               chunk_size: int,
               ) -> torch.Tensor:
        mha_inputs = {
            "q_x": z,
            "kv_x": t,
            "biases": biases,
        }
        return chunk_layer(
            self.mha,
            mha_inputs,
            chunk_size=chunk_size,
            no_batch_dims=len(z.shape[:-2]),
        )

    def forward(self,
                t: torch.Tensor,
                z: torch.Tensor,
                template_mask: Optional[torch.Tensor] = None,
                chunk_size: Optional[int] = 256
                ) -> torch.Tensor:
        """
        Note that this module suffers greatly from a small chunk size

        Args:
            t:
                [*, N_templ, N_res, N_res, C_t] template embedding
            z:
                [*, N_res, N_res, C_t] pair embedding
            template_mask: [*, N_templ] template mask
        Returns:
            [*, N_res, N_res, C_z] pair embedding update
        """

        # [*, N_res, N_res, 1, C_z]
        z = z.unsqueeze(-2)
        # [*, N_res, N_res, N_temp, C_t]
        t = permute_final_dims(t, (1, 2, 0, 3))
        # [*, N_res, N_res, 1, C_z]
        biases = []
        if template_mask is not None:
            bias = self.inf * (template_mask[..., None, None, None, None, :] - 1)
            biases.append(bias)

        if chunk_size is not None and not self.training:
            z = self._chunk(z, t, biases, chunk_size)
        else:
            z = self.mha(q_x=z, kv_x=t, biases=biases)

        # [*, N_res, N_res, C_z]
        z = z.squeeze(-2)

        return z


class TemplatePairBlock(nn.Module):
    def __init__(
            self,
            c_t: int,
            c_hidden_tri_mul: int,
            num_heads: int = 4,
            pair_transition_n: int = 2,
            dropout_rate: float = 0.0,
            inf: float = 1e5
    ):
        super(TemplatePairBlock, self).__init__()
        self.c_t = c_t
        self.c_hidden_tri_mul = c_hidden_tri_mul
        self.num_heads = num_heads
        self.pair_transition_n = pair_transition_n
        self.dropout_rate = dropout_rate
        self.inf = inf
        self.dropout_row = DropoutRowwise(self.dropout_rate)
        self.dropout_col = DropoutColumnwise(self.dropout_rate)
        self.tri_att_start = TriangleAttentionStartingNode(
            self.c_t,
            self.num_heads,
            inf=inf
        )
        self.tri_att_end = TriangleAttentionEndingNode(
            self.c_t,
            self.num_heads,
            inf=inf,
        )
        self.tri_mul_out = TriangleMultiplicationOutgoing(
            self.c_t,
            self.c_hidden_tri_mul
        )
        self.tri_mul_in = TriangleMultiplicationIncoming(
            self.c_t,
            self.c_hidden_tri_mul
        )
        self.pair_transition = Transition(self.c_t, self.pair_transition_n)

    def forward(self,
                z: torch.Tensor,
                mask: torch.Tensor = None,
                chunk_size: Optional[int] = None
                ):
        single_templates = [
            t.unsqueeze(-4) for t in torch.unbind(z, dim=-4)
        ]

        if mask is not None:
            single_templates_masks = [
                m.unsqueeze(-3) for m in torch.unbind(mask, dim=-3)
            ]
        else:
            single_templates_masks = [None, ] * len(single_templates)

        for i in range(len(single_templates)):
            single = single_templates[i]
            single_mask = single_templates_masks[i]
            single = single + self.dropout_row(
                self.tri_att_start(single, mask=single_mask)
            )
            single = single + self.dropout_col(self.tri_att_end(single, mask=single_mask))
            tmu_update = self.tri_mul_out(
                single,
                mask=single_mask
            )
            single = single + self.dropout_row(tmu_update)
            tmu_update = self.tri_mul_in(
                single,
                mask=single_mask
            )
            single = single + self.dropout_row(tmu_update)
            single = single + self.pair_transition(single, chunk_size=chunk_size)
            single_templates[i] = single

        z = torch.cat(single_templates, dim=-4)

        return z


class TemplatePairStack(nn.Module):
    """Implements Algorithm 16.

    Args:
        c_t:
            Template embedding channel dimension
        c_hidden_tri_att:
            Hidden dimension for triangular multiplication
        num_blocks:
            Number of blocks in the stack
        pair_transition_n:
            Scale of pair transition (Alg. 15) hidden dimension
        dropout_rate:
            Dropout rate used throughout the stack
    """

    def __init__(
            self,
            c_t,
            c_hidden_tri_mul=64,
            num_blocks=2,
            num_heads=4,
            pair_transition_n=2,
            dropout_rate=0.25,
            inf=1e9
    ):
        super(TemplatePairStack, self).__init__()
        self.blocks_per_ckpt = False
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            block = TemplatePairBlock(
                c_t=c_t,
                c_hidden_tri_mul=c_hidden_tri_mul,
                num_heads=num_heads,
                pair_transition_n=pair_transition_n,
                dropout_rate=dropout_rate,
                inf=inf,
            )
            self.blocks.append(block)
        self.layer_norm = LayerNorm(c_t)
        self.activation_checkpoint = False

    def forward(
            self,
            t: torch.tensor,
            mask: torch.tensor = None,
            chunk_size: int = None
    ):
        """
        Args:
            t:
                [*, N_templ, N_res, N_res, C_t] template embedding
            mask:
                [*, N_templ, N_res, N_res] mask
        Returns:
            [*, N_templ, N_res, N_res, C_t] template embedding update
        """
        if mask is not None and mask.shape[-3] == 1:
            expand_idx = list(mask.shape)
            expand_idx[-3] = t.shape[-4]
            mask = mask.expand(*expand_idx)

        blocks = [
            partial(
                b,
                mask=mask,
                chunk_size=chunk_size
            )
            for b in self.blocks
        ]
        if chunk_size is not None:
            blocks = [partial(b, chunk_size=chunk_size) for b in blocks]

        t, = checkpoint_blocks(
            blocks=blocks,
            args=(t,),
            interval=self.activation_checkpoint if self.training else None,
        )

        t = self.layer_norm(t)

        return t


class TemplateEmbedding(nn.Module):

    def __init__(self, config):
        self.config = config
        super(TemplateEmbedding, self).__init__()

        self.template_pair_stack = TemplatePairStack(
            **self.config["template_pair_stack"]
        )
        self.template_pointwise_att = TemplatePointwiseAttention(
            **self.config["template_pointwise_attention"]
        )

    def forward(self,
                batch,
                z,
                pair_mask=None,
                chunk_size=None):
        template_mask = batch.get("template_mask", None)
        if template_mask is not None:
            template_mask.to(dtype=z.dtype)

        t_pair_feats = batch["template_pair_feats"]  # [*, N, N, 88]
        t_pair = self.template_pair_embedder(t_pair_feats)
        # [*, S_t, N, N, C_z]
        t = self.template_pair_stack(
            t_pair,
            mask=pair_mask.unsqueeze(-3) if pair_mask is not None else None,
            chunk_size=chunk_size
        )
        # [*, N, N, C_z]
        t = self.template_pointwise_att(t, z, template_mask=template_mask)
        if template_mask is not None:
            t_mask = (torch.sum(template_mask, dim=-1) > 0).to(t.dtype)
            # Append singletons
            t_mask = t_mask.reshape(*t_mask.shape, *([1] * (len(t.shape) - len(t_mask.shape))))
            t = t * t_mask

        ret = {"template_pair_embedding": t}
        if self.template_angle_enabled:
            # [*, S_t, N, C_m]
            ret["template_angle_embedding"] = self.template_angle_embedder(batch["template_angle_feat"])

        return ret

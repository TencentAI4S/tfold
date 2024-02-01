# -*- coding: utf-8 -*-
# Copyright (c) 2023, Tencent Inc. All rights reserved.
import torch
import torch.nn as nn

from tfold.model.layer import get_activation_fn


# TODO: change to standord layernorm
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, pb_relax=False):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
        self.pb_replax = pb_relax

    def forward(self, x):
        if self.pb_replax:
            x = x / (x.abs().max().detach() / 8)

        mean = x.mean(-1, keepdim=True)
        std = torch.sqrt(x.var(dim=-1, keepdim=True, unbiased=False) + self.eps)
        x = self.a_2 * (x - mean)
        x /= std
        x += self.b_2
        return x


class MLP(nn.Module):
    def __init__(self,
                 dim,
                 ffn_dim,
                 p_drop=0.1,
                 d_model_out=None,
                 activation='relu',
                 is_post_act_ln=False,
                 pb_relax=False
                 ):
        super(MLP, self).__init__()
        d_model_out = d_model_out or dim
        self.linear1 = nn.Linear(dim, ffn_dim)
        self.post_act_ln = LayerNorm(ffn_dim, pb_relax=pb_relax) if is_post_act_ln else nn.Identity()
        self.dropout = nn.Dropout(p_drop)
        self.linear2 = nn.Linear(ffn_dim, d_model_out)
        self.activation = get_activation_fn(activation)

    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.post_act_ln(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

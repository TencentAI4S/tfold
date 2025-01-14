# Copyright (c) 2024, Tencent Inc. All rights reserved.
# Data: 2024/5/29 17:47
from functools import partial
from typing import Optional

import torch
import torch.nn as nn

from tfold.model.layer import LayerNorm, Linear
from tfold.model.utils import chunk_layer


class OuterProductMeanMSA(nn.Module):
    """Outer-produce mean for transforming sequence features into an update for pair features.
    Implements Algorithm 10.

    Args:
        c_m: MSA embedding channel dimension
        c_z: Pair embedding channel dimension
        c_hidden: Hidden channel dimension
    """

    def __init__(self, c_m, c_z, c_hidden=32, eps=1e-3):
        super(OuterProductMeanMSA, self).__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.eps = eps
        self.layer_norm = LayerNorm(c_m)
        self.linear_1 = Linear(c_m, c_hidden)
        self.linear_2 = Linear(c_m, c_hidden)
        self.linear_out = Linear(c_hidden ** 2, c_z, init="final")

    def _opm(self, a, b):
        # [*, N_res, N_res, C, C]
        outer = torch.einsum("...bac,...dae->...bdce", a, b)
        # [*, N_res, N_res, C * C]
        outer = outer.reshape(outer.shape[:-2] + (-1,))
        # [*, N_res, N_res, C_z]
        outer = self.linear_out(outer)

        return outer

    @torch.jit.ignore
    def _chunk(self,
               a: torch.Tensor,
               b: torch.Tensor,
               chunk_size: int
               ) -> torch.Tensor:
        a_reshape = a.reshape((-1,) + a.shape[-3:])
        b_reshape = b.reshape((-1,) + b.shape[-3:])
        out = []
        for a_prime, b_prime in zip(a_reshape, b_reshape):
            outer = chunk_layer(
                partial(self._opm, b=b_prime),
                {"a": a_prime},
                chunk_size=chunk_size,
                no_batch_dims=1,
            )
            out.append(outer)

        # For some cursed reason making this distinction saves memory
        if len(out) == 1:
            outer = out[0].unsqueeze(0)
        else:
            outer = torch.stack(out, dim=0)

        outer = outer.reshape(a.shape[:-3] + outer.shape[1:])

        return outer

    def forward(self,
                m: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                chunk_size: Optional[int] = None
                ) -> torch.Tensor:
        """
        Args:
            m: [*, num_seqs, seq_len, c_m] MSA embedding
            mask: [*, num_seqs, seq_len] MSA mask

        Returns:
            [*, seq_len, seq_len, C_z] pair embedding update
        """
        ln = self.layer_norm(m)
        a = self.linear_1(ln)
        b = self.linear_2(ln)
        if mask is not None:
            mask = mask.unsqueeze(-1)
            a = a * mask
            b = b * mask

        a = a.transpose(-2, -3)
        b = b.transpose(-2, -3)

        if chunk_size is not None:
            outer = self._chunk(a, b, chunk_size)
        else:
            outer = self._opm(a, b)

        if mask is None:
            # [*, num_seqs, seq_len]
            mask = m.new_ones(m.shape[:-1] + (1,))

        norm = torch.einsum("...abc,...adc->...bdc", mask, mask)
        outer = outer / (norm + self.eps)

        return outer


class OuterProductMeanSM(nn.Module):
    """
    Outer-produce mean for update complex pair feature using
    antibody single feature and antigen MSA feature

    Args:
        c_s:
            Sequence embedding channel dimension (antibody)
        c_m:
            MSA embedding channel dimension (antigen)
        c_z:
            Pair embedding channel dimension
        c_hidden:
            Hidden channel dimension
    """

    def __init__(self, c_s, c_m, c_z, c_hidden=32, eps=1e-3):
        super(OuterProductMeanSM, self).__init__()
        self.c_s = c_s
        self.c_m = c_m
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.eps = eps

        self.layer_norm_ab = LayerNorm(c_s)
        self.layer_norm_ag = LayerNorm(c_m)

        self.linear_ab = Linear(c_s, c_hidden)
        self.linear_ag = Linear(c_m, c_hidden)
        self.linear_out = Linear(c_hidden ** 2, c_z)

    def _opm(self, a, b):
        K = b.shape[2]
        a = a.unsqueeze(dim=2).repeat(1, 1, K, 1)  # a: N x L_1 x c -> N x L_1 x K x c
        outer = torch.einsum('...bac,...dae->...bdce', a, b)  # N x L_1 x L_2 x c x c
        outer = outer.reshape(outer.shape[:3] + (-1,))
        outer = self.linear_out(outer)  # N x L_1 x L_2 x c_z

        return outer

    def forward(self, s, m):
        """
        Args:
            s: antibody single feature of size N x L_1 x c_s
            m: antigen MSA feature of size N x K x L_2 x c_s

        Returns:
            z: update term for antidiagonal pair features of size N x L_1 x L_2 x c_z
        """

        K = m.shape[1]
        smsk_tns = s.new_ones(s.shape[:-1]).unsqueeze(-1)  # N x L_1 x 1
        smsk_tns = smsk_tns.unsqueeze(1).repeat(1, K, 1, 1)  # N x K x L_1 x 1
        mmsk_tns = m.new_ones(m.shape[:-1]).unsqueeze(-1)  # N x K x L_2 x 1

        s = self.layer_norm_ab(s)
        s = self.linear_ab(s)

        m = self.layer_norm_ag(m)
        m = self.linear_ag(m)
        m = m.transpose(-2, -3)  # N x L_2 x K x c

        z = self._opm(s, m)
        norm = torch.einsum('...abc,...adc->...bdc', smsk_tns, mmsk_tns)
        norm = norm + self.eps

        return z / norm


class OuterProductMeanSS(nn.Module):
    """Outer-produce mean for transforming single features into an update for pair features."""

    def __init__(self, c_s, c_z, c_hidden=32):
        super(OuterProductMeanSS, self).__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.layer_norm = LayerNorm(c_s)
        self.linear_1 = Linear(c_s, c_hidden)
        self.linear_2 = Linear(c_s, c_hidden)
        self.linear_out = Linear(c_hidden ** 2, c_z, init='final')

    def forward(self, s):
        """
        Args:
            s: [N, L, c_s], single features

        Returns:
            z: update term for pair features of size N x L x L x c_z
        """

        N, L, _ = s.shape  # pylint: disable=invalid-name
        s = self.layer_norm(s)
        afea_tns = self.linear_1(s).view(N, L, 1, self.c_hidden, 1)
        bfea_tns = self.linear_2(s).view(N, 1, L, 1, self.c_hidden)
        ofea_tns = (afea_tns * bfea_tns).view(N, L, L, self.c_hidden ** 2)

        return self.linear_out(ofea_tns)

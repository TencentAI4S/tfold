"""The Evoformer block for single-sequence inputs."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from tfold.model.layer import (
    Linear,
    LayerNorm,
    DropoutRowwise,
    DropoutColumnwise,
    Attention,
    TriangleMultiplicationOutgoing,
    TriangleMultiplicationIncoming,
    TriangleAttentionStartingNode,
    TriangleAttentionEndingNode
)


class OuterProductMean(nn.Module):
    """Outer-produce mean for transforming single features into an update for pair features."""

    def __init__(self, c_s, c_z, c_h=32):
        super(OuterProductMean, self).__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_h = c_h
        self.layer_norm = LayerNorm(c_s)
        self.linear_1 = Linear(c_s, c_h)
        self.linear_2 = Linear(c_s, c_h)
        self.linear_out = Linear(c_h ** 2, c_z, init='final')

    def forward(self, s):
        """
        Args:
            s: [N, L, c_s], single features

        Returns:
            z: update term for pair features of size N x L x L x c_z
        """

        N, L, _ = s.shape  # pylint: disable=invalid-name
        s = self.layer_norm(s)
        afea_tns = self.linear_1(s).view(N, L, 1, self.c_h, 1)
        bfea_tns = self.linear_2(s).view(N, 1, L, 1, self.c_h)
        ofea_tns = (afea_tns * bfea_tns).view(N, L, L, self.c_h ** 2)

        return self.linear_out(ofea_tns)


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


class SeqAttentionWithPairBias(nn.Module):
    """Sequence attention with pairwise biases.

    Args:
        c_s: number of dimensions in single features
        c_z: number of dimensions in pair features
        c_h: number of dimensions in query/key/value embeddings
        num_heads: number of attention heads
    """

    def __init__(self, c_s, c_z, c_h=32, num_heads=12):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_h = c_h
        self.num_heads = num_heads
        self.layer_norm_s = LayerNorm(self.c_s)
        self.layer_norm_z = LayerNorm(self.c_z)
        self.linear_z = Linear(self.c_z, self.num_heads, bias=False, init='normal')
        self.mha = Attention(self.c_s, self.c_s, self.c_s, self.c_h, self.num_heads)

    def forward(self, s, z):
        """
        Args:
            s: [N, L, c_s], single features
            z: [N, L, L, c_z]pair features

        Returns:
            s: [N, L, c_s], updated single features
        """

        s = self.layer_norm_s(s)
        z = self.layer_norm_z(z)
        bias_tns = self.linear_z(z).permute(0, 3, 1, 2)
        s = self.mha(s, s, [bias_tns])

        return s


class Transition(nn.Module):
    """Transition module for both single & pair features."""

    def __init__(self, c, n=4):
        super().__init__()
        self.c = c
        self.n = n
        self.layer_norm = LayerNorm(self.c)
        self.linear_1 = Linear(self.c, self.n * self.c)
        self.relu = nn.ReLU()
        self.linear_2 = Linear(self.n * self.c, c)

    def forward(self, x):
        """
        Args:
            x: [N, L, c] or [N, L, L, c], single/pair features

        Returns:
            x: updated single/pair features
        """
        x = self.layer_norm(x)
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)

        return x


class EvoformerBlockSS(nn.Module):
    """The Evoformer block for single-sequence inputs."""

    def __init__(
            self,
            c_s: int,
            c_z: int,
            c_h_seq_att: int = 32,
            c_h_opm: int = 32,
            c_h_pair_mul: int = 128,
            c_h_pair_att: int = 32,
            n_heads_seq: int = 12,
            n_heads_pair: int = 8,
            dropout_seq: float = 0.15,
            dropout_pair: float = 0.25,
            inf: float = 1e9
    ):
        super().__init__()
        # single stack
        self.seq_att = SeqAttentionWithPairBias(c_s, c_z, c_h_seq_att, n_heads_seq)
        self.seq_trans = Transition(c_s)

        # single to pair
        self.opm = OuterProductMean(c_s, c_z, c_h_opm)

        # pair stack
        self.tri_mul_out = TriangleMultiplicationOutgoing(c_z, c_h_pair_mul)
        self.tri_mul_in = TriangleMultiplicationIncoming(c_z, c_h_pair_mul)
        self.tri_att_start = TriangleAttentionStartingNode(c_z, c_h_pair_att, n_heads_pair, inf=inf)
        self.tri_att_end = TriangleAttentionEndingNode(c_z, c_h_pair_att, n_heads_pair, inf=inf)
        self.pair_trans = Transition(c_z)

        # dropout
        self.seq_dropout = DropoutRowwise(dropout_seq)
        self.pair_dropout_row = DropoutRowwise(dropout_pair)
        self.pair_dropout_col = DropoutColumnwise(dropout_pair)

    def forward(self, s, z, chunk_size=None):
        """
        Args:
            s: [N, L, c_s], single features
            z: [N, L, L, c_z], pair features

        Returns:
            s: [N, L, c_s], updated single features
            z: [N, L, L, c_z], updated pair features
        """

        # single stack
        s = s + self.seq_dropout(self.seq_att(s, z))
        s = s + self.seq_trans(s)

        # single to pair
        z = z + self.opm(s)

        # pair stack
        z = z + self.pair_dropout_row(self.tri_mul_out(z))
        z = z + self.pair_dropout_row(self.tri_mul_in(z))
        z = z + self.pair_dropout_row(self.tri_att_start(z, chunk_size=chunk_size))
        z = z + self.pair_dropout_col(self.tri_att_end(z, chunk_size=chunk_size))
        z = z + self.pair_trans(z)

        return s, z


class EvoformerStackSS(nn.Module):
    """Stacked EvoformerBlockSS layers.

    Args:
        c_s: number of dimensions in single features
        c_z: number of dimensions in pair features
        num_layers: number of EvoformerBlockSS layers
    """

    def __init__(
            self,
            c_s=384,
            c_z=256,
            num_layers=8,  # number of <EvoformerBlockSS> layers
            use_checkpoint=True,  # whether to use the checkpoint mechanism to avoid OMM
    ):
        super().__init__()
        self.num_layers = num_layers
        self.c_s = c_s
        self.c_z = c_z
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            EvoformerBlockSS(self.c_s, self.c_z)
            for _ in range(self.num_layers)
        ])

    def forward(self, s, z, num_recycles=1, chunk_size=None):
        """
        Args:
            s: [N x L x c_s], single features
            z: pair features of size N x L x L x c_z
            num_recycles: (optional) number of recycling iterations

        Returns:
            s: updated single features of size N x L x c_s
            z: updated pair features of size N x L x L x c_z
        """
        for _ in range(num_recycles):
            for block in self.blocks:
                if not (self.training and self.use_checkpoint):
                    s, z = block(s, z, chunk_size)
                else:
                    s, z = checkpoint(block, s, z, chunk_size)
        return s, z

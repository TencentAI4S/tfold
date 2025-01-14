# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
import numpy as np
import torch
import torch.nn as nn


class OnehotEmbedding(nn.Module):
    """One-hot encoder."""

    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab

    @property
    def n_dims(self):
        """Get the number of dimensions needed for one-hot encodings."""
        return len(self.vocab)

    def name2idx(self, names):
        """Convert names into indices.

        Args:
            names: list of names of length L

        Returns:
            idxs_vec: indices of size L
        """

        idxs_vec = torch.tensor([self.vocab.index(x) for x in names], dtype=torch.int64)

        return idxs_vec

    def name2onht(self, names):
        """Convert names into one-hot encodings.

        Args:
            names: list of names of length L

        Returns:
            onht_mat: one-hot encodings of size L x D
        """

        idxs_vec = self.name2idx(names)
        onht_mat = nn.functional.one_hot(idxs_vec, self.n_dims)

        return onht_mat

    def idx2name(self, idxs_vec):
        """Convert indices into names.

        Args:
            idxs_vec: indices of size L

        Returns:
            names: list of names of length L
        """

        idxs_vec_np = idxs_vec.detach().cpu().numpy()
        names = [self.vocab[x] for x in np.nditer(idxs_vec_np)]

        return names

    def onht2name(self, onht_mat):
        """Convert one-hot encodings into residue names.

        Args:
            onht_mat: one-hot encodings of size L x D

        Returns:
            names: list of names of length L

        Note:
        * It is also okay to feed predicted probabilities for conversion.
        """

        idxs_vec = torch.argmax(onht_mat, dim=1)
        names = self.idx2name(idxs_vec)

        return names

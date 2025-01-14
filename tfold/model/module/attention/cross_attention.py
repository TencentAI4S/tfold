# Copyright (c) 2024, Tencent Inc. All rights reserved.
# Data: 2024/5/29 15:45
import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attention1 = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.attention2 = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, sfea_tns1, sfea_tns2):
        """
        Args:
            sfea_tns1: sequential feature_1 of size N x L1 x c_s
            sfea_tns2: sequential feature_2 of size N x L2 x c_s

        Returns:
            sfea_tns: merged sequential feature of size N x (L1+L2) x c_s
        """
        attn1, _ = self.attention1(sfea_tns1, sfea_tns2, sfea_tns2)
        attn2, _ = self.attention2(sfea_tns2, sfea_tns1, sfea_tns1)

        sfea_tns = torch.cat([attn1, attn2], dim=1)

        return sfea_tns

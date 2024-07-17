import math

import torch
from torch import nn

from src.timeseries_transformer.utils import clone_layers


class MultiHeadAttention(nn.Module):
    def __init__(self, head_size, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert head_size % num_heads == 0

        self.d_k = head_size // num_heads
        self.weight_matrices = clone_layers(nn.Linear(head_size, head_size), num_heads)
        self.attn = None
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)

    def _attention(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        # if mask is not None:
        #   scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = scores.softmax(dim=-1)
        # if dropout is not None:
        #   p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def forward(self, query, key, value):
        """Perform forward pass of MultiheadAttention module."""
        # get q, k and v
        query, key, value = [
          weights(inputs)
          for weights, inputs in zip(self.weight_matrices, (query, key, value))
        ]

        # calculate attention
        x, self.attn = self._attention(query, key, value)

        return self.weight_matrices[-1](x)
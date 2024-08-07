"""Module containing different attention mechanisms."""
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


class SparseAttention(nn.Module):
    def __init__(self, head_size, num_heads, sparsity, dropout=0.1):
        super(SparseAttention, self).__init__()
        assert head_size % num_heads == 0
        self.d_k = head_size // num_heads
        self.num_heads = num_heads
        self.sparsity = sparsity
        self.weight_matrices = clone_layers(nn.Linear(head_size, head_size), num_heads)
        self.attn = None
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)

    def _sparse_attention(self, query, key, value, mask=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply sparsity mask
        sparse_mask = torch.rand_like(scores) > self.sparsity
        scores = scores.masked_fill(~sparse_mask, -float('inf'))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -float('inf'))

        p_attn = scores.softmax(dim=-1)

        if hasattr(self, 'dropout'):
            p_attn = self.dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query, key, value = [
            weights(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            for weights, x in zip(self.weight_matrices, (query, key, value))
        ]

        x, self.attn = self._sparse_attention(query, key, value, mask)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        return self.weight_matrices[-1](x)


class WindowedAttention(nn.Module):
    def __init__(self, head_size, num_heads, window_size, dropout=0.1):
        super(WindowedAttention, self).__init__()
        assert head_size % num_heads == 0
        self.d_k = head_size // num_heads
        self.num_heads = num_heads
        self.window_size = window_size
        self.weight_matrices = clone_layers(nn.Linear(head_size, head_size), num_heads)
        self.attn = None
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)

    def _windowed_attention(self, query, key, value, mask=None):
        batch_size, num_heads, seq_len, d_k = query.size()

        # Create a windowed version of the key and value tensors
        key_windowed = key.unfold(2, self.window_size, 1)
        value_windowed = value.unfold(2, self.window_size, 1)

        # Compute attention scores
        scores = torch.matmul(query.unsqueeze(3), key_windowed.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            window_mask = mask.unfold(1, self.window_size, 1)
            scores = scores.masked_fill(window_mask.unsqueeze(1).unsqueeze(2) == 0, -float('inf'))

        p_attn = scores.softmax(dim=-1)

        if hasattr(self, 'dropout'):
            p_attn = self.dropout(p_attn)

        # Apply attention to windowed values
        x = torch.matmul(p_attn, value_windowed)

        return x.squeeze(3), p_attn

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query, key, value = [
            weights(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            for weights, x in zip(self.weight_matrices, (query, key, value))
        ]

        x, self.attn = self._windowed_attention(query, key, value, mask)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        return self.weight_matrices[-1](x)
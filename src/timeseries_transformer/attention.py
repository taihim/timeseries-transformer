"""Module containing different attention mechanisms."""
import math

import torch
from torch import nn

from src.timeseries_transformer.utils import clone_layers


class ScaledDPAttention(nn.Module):
    def __init__(self, head_size, dropout=0.1):
        super(ScaledDPAttention, self).__init__()
        self.d_k = head_size
        self.proj = nn.Linear(head_size, head_size * 3)  # Combined projection for Q, K, V
        self.attn = None
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)

    def _attention(self, query, key, value, mask=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = scores.softmax(dim=-1)
        if hasattr(self, 'dropout'):
            p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def forward(self, x, mask=None):
        """Perform forward pass of SingleAttention module."""
        # Combined projection for Q, K, V
        qkv = self.proj(x).chunk(3, dim=-1)
        query, key, value = qkv

        # Calculate attention
        x, self.attn = self._attention(query, key, value, mask)

        return x


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

        assert seq_len % self.window_size == 0

        # Create a windowed version of the key and value tensors
        key_windowed = key.unfold(2, self.window_size, self.window_size)
        value_windowed = value.unfold(2, self.window_size, self.window_size)

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


class LinearAttention(nn.Module):
    def __init__(self, head_size, num_heads, seq_len, low_rank_dim, dropout=0.1):
        super(LinearAttention, self).__init__()
        assert head_size % num_heads == 0
        self.d_k = head_size // num_heads
        self.num_heads = num_heads
        self.low_rank_dim = low_rank_dim

        self.weight_matrices = clone_layers(nn.Linear(head_size, head_size), num_heads)
        self.E = nn.Parameter(torch.Tensor(num_heads, seq_len, low_rank_dim))
        self.F = nn.Parameter(torch.Tensor(num_heads, seq_len, low_rank_dim))

        self.attn = None
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)

        # Initialize E and F
        nn.init.xavier_uniform_(self.E)
        nn.init.xavier_uniform_(self.F)

    def _linear_attention(self, query, key, value):
        # Project key and value
        key_projected = torch.matmul(key.transpose(-2, -1), self.E)
        value_projected = torch.matmul(value.transpose(-2, -1), self.F)

        # Compute attention
        attn = torch.matmul(query, key_projected) / math.sqrt(self.d_k)
        attn = attn.softmax(dim=-1)

        if hasattr(self, 'dropout'):
            attn = self.dropout(attn)

        # Apply attention to value
        output = torch.matmul(attn, value_projected.transpose(-2, -1))
        return output, attn

    def forward(self, query, key, value):
        batch_size = query.size(0)

        # Linear projections
        query, key, value = [
            weights(inputs).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            for weights, inputs in zip(self.weight_matrices, (query, key, value))
        ]

        # Apply linear attention
        x, self.attn = self._linear_attention(query, key, value)

        # Concatenate heads and apply final linear projection
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.weight_matrices[-1](x)
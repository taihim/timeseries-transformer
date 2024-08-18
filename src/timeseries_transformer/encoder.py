"""Encoder module for the timeseries transformer model."""
from torch import nn
from src.timeseries_transformer.attention import MultiHeadAttention, WindowedAttention, LinearAttention


class PytorchEncoder(nn.Module):
    def __init__(self, input_shape, embed_size, num_heads, ff_dim, dropout=0):
        super(PytorchEncoder, self).__init__()
        # attention
        self.embedding = nn.Linear(in_features=input_shape[-1], out_features=embed_size)
        self.attention = LinearAttention(embed_size, num_heads, dropout=0.0, seq_len=500, low_rank_dim=32)
        # self.attention = MultiHeadAttention(embed_size, num_heads, dropout=0.0)
        self.linear1 = nn.Linear(embed_size, 1)
        self.dropout1 = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(normalized_shape=input_shape[-1], eps=1e-6)

        # feedforward
        self.conv1 = nn.Conv1d(in_channels=input_shape[-1], out_channels=ff_dim, kernel_size=1)
        self.relu1 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(in_channels=ff_dim, out_channels=input_shape[-1], kernel_size=1)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=input_shape[1], eps=1e-6)

    def forward(self, src):
        """Perform forward pass of the encoder module."""
        x = self.embedding(src)
        x = self.attention(x, x, x)[0]
        x = self.linear1(x)
        x = self.dropout1(x)
        x = self.layer_norm1(x)

        res = x + src
        res = res.reshape(res.shape[0], res.shape[2], res.shape[1])

        x = self.conv1(res)
        x = self.relu1(x)
        x = self.dropout2(x)
        x = self.conv2(x)
        x = self.layer_norm2(x)
        x = x + res

        return x.reshape(x.shape[0], x.shape[-1], x.shape[1])
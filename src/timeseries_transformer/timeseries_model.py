"""Timeseries transformer model for classification tasks."""
from collections import OrderedDict
import torch.nn as nn
import torch

from src.timeseries_transformer.constants import MLP_UNITS
from src.timeseries_transformer.encoder import PytorchEncoder


class EncoderClassifier(nn.Module):
    def __init__(self, input_shape, embed_size, num_heads, ff_dim, dropout=0, num_blocks=2):
        super(EncoderClassifier, self).__init__()

        encoder_layer = PytorchEncoder(input_shape=input_shape, embed_size=embed_size, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout)
        encoders = OrderedDict()
        for idx in range(num_blocks):
            encoders[f"encoder{idx}"] = encoder_layer

        self.encoder_block = nn.Sequential(encoders)
        self.avg = nn.AvgPool1d(kernel_size=1)
        self.dense1 = nn.Linear(input_shape[1], MLP_UNITS[0])
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.dense2 = nn.Linear(MLP_UNITS[0], 2)
        self.softmax = nn.Softmax()

    def forward(self, x: torch.Tensor):
        """Perform forward pass of the classifier model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, num_features).
        """
        x = self.encoder_block(x)
        x = torch.squeeze(self.avg(x), 2)
        x = self.dense1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.softmax(x)
        return x

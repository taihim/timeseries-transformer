"""Utility functions for the timeseries_transformer module."""
from src.timeseries_transformer.datasets import DatasetBuilder
from torch.utils.data import DataLoader
from torch import nn
import copy


def clone_layers(module, n):
    "Produce n independent but identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


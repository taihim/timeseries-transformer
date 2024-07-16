"""Utility functions for the timeseries_transformer module."""
from torch import nn, save as torch_save
import copy


def clone_layers(module, n):
    """Produce n independent but identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def save_model(model, path):
    """Save model to disk."""
    torch_save(model.state_dict(), path)


def save_plots():
    """Save plots to disk."""
    pass

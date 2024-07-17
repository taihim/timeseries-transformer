"""Utility functions for the timeseries_transformer module."""
import os
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import classification_report
import torch
from torch import nn, save as torch_save
import copy

from datetime import datetime

from src.timeseries_transformer.constants import BASE_PATH_RESULTS


def clone_layers(module, n):
    """Produce n independent but identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def evaluate_model(model, data_loader):
    """Evaluate the model on the given data."""
    acc = 0
    iteration = 0
    model["model"].eval()
    for data in data_loader:
        iteration += 1
        inputs, labels = data
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        outputs = model["model"](inputs)
        predictions = torch.argmax(outputs, dim=1)
        correct_labels = labels.squeeze().int()

        acc += (predictions == correct_labels).int().sum()/len(labels) * 100
    x = classification_report(correct_labels, predictions)
    print(f"Evaluation result for model {model["id"]}: {acc/iteration}")


def save_model_and_results(
        model: list[Any],
        losses: float | list[float],
        accuracies: float | list[float],
        epoch_times: float | list[float],
        eval_results: dict[str, float] | list[dict[str, float]]):
    """Save model to disk."""
    current_date = datetime.now().strftime('%Y-%m-%d')

    os.makedirs(Path(f"{BASE_PATH_RESULTS}/{current_date}/{model["id"]}"), exist_ok=True)
    numpy_loss = [np.array(loss_tuple) for loss_tuple in losses]
    numpy_acc = [np.array(acc_tuple) for acc_tuple in accuracies]

    torch_save(model["model"].state_dict(), Path(f"{BASE_PATH_RESULTS}/{current_date}/{model["id"]}/{model["id"]}.pt"))
    np.savez(Path(f"{BASE_PATH_RESULTS}/{current_date}/{model["id"]}/{model["id"]}_losses.npz"), *numpy_loss)
    np.savez(Path(f"{BASE_PATH_RESULTS}/{current_date}/{model["id"]}/{model["id"]}_accuracies.npz"), *numpy_acc)

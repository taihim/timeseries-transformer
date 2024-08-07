"""Utility functions for the timeseries_transformer module."""
import os
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
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
    all_predictions = []
    all_correct_labels = []
    for data in data_loader:
        iteration += 1
        inputs, labels = data
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        outputs = model["model"](inputs)
        predictions = torch.argmax(outputs, dim=1)
        correct_labels = labels.squeeze().int()
        all_predictions.extend(predictions.cpu())
        all_correct_labels.extend(correct_labels.cpu())

        acc += (predictions == correct_labels).int().sum()/len(labels) * 100

    print(f"Evaluation accuracy for model {model["id"]}: {acc / iteration}")
    report = classification_report(all_correct_labels, all_predictions)
    conf_matrix = confusion_matrix(all_correct_labels, all_predictions)
    # TODO: save these (report, conf_matrix, acc/iteration (as test set accuracy)) to disk (maybe as csv or json?)
    print(report)
    print(conf_matrix)



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

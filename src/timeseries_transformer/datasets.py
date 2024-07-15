from torch.utils.data import Dataset
import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold

class FordDataset(Dataset):
    def __init__(self, sequences, labels):
        self.labels = labels
        self.sequences = sequences
        self.num_classes = len(torch.unique(self.labels))  # count the number of unique labels

    def __len__(self):
        return self.sequences.shape[0]

    def __getitem__(self, idx):
        sequence = torch.reshape(self.sequences[idx], (-1, 1))  # dim: seq_len x num_features
        label = torch.reshape(self.labels[idx], (-1,))  # dim: 1 x 1

        return sequence, label


class FordDataset(Dataset):
    def __init__(self, split="train"):
        self.root_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"
        self.data = torch.tensor(np.loadtxt(self.root_url + "FordA_TRAIN.tsv", delimiter="\t"),
                                 dtype=torch.float32) if split == "train" else torch.tensor(
            np.loadtxt(self.root_url + "FordA_TEST.tsv", delimiter="\t"), dtype=torch.float32)
        self.labels = self.data[:, 0]  # get first element from each example
        self.sequences = self.data[:, 1:]  # get all elements after first element
        self.labels[self.labels == -1] = 0  # change all -1 labels to 0
        self.num_classes = len(torch.unique(self.labels))  # count the number of unique labels

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sequence = torch.reshape(self.sequences[idx], (-1, 1))  # dim: seq_len x num_features
        label = torch.reshape(self.labels[idx], (-1,))  # dim: 1 x 1

        return sequence, label


class DatasetManager:
    def __init__(self, split: str = "train", use_k_fold: bool = False, num_folds: None | int = None) -> None:
        self.root_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"
        self.split = split

        if self.split == "train":
            self.raw_data = torch.tensor(np.loadtxt(self.root_url + "FordA_TRAIN.tsv", delimiter="\t"), dtype=torch.float32)
        else:
            self.raw_data = torch.tensor(np.loadtxt(self.root_url + "FordA_TEST.tsv", delimiter="\t"), dtype=torch.float32)

        if use_k_fold:
            self.use_k_fold = use_k_fold
            self.num_folds = num_folds
            self.skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

        self.labels = self.raw_data[:, 0]  # get first element from each example
        self.sequences = self.raw_data[:, 1:]  # get all elements after first element
        self.labels[self.labels == -1] = 0  # change all -1 labels to 0
        self.num_classes = len(torch.unique(self.labels))  # count the number of unique labels


    def get_dataset(self) -> Dataset | list[Dataset]:
        if self.use_k_fold:
            for train_idx, val_idx in self.skf.split(self.sequences, self.labels):
                # Create PyTorch DataLoader for training and validation
                train_sets.append(FordDatasetKfold(x[train_idx], y[train_idx]))
                val_sets.append(FordDatasetKfold(x[val_idx], y[val_idx]))
            return []
        return FordDataset(self.sequences, self.labels)
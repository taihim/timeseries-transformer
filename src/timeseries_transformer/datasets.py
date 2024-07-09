import torch.data.Dataset
from torch.utils.data import Dataset
import torch
import numpy as np


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

from torch.utils.data import Dataset
import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

class FordDataset(Dataset):
    """Dataset class for FordA dataset. The dataset is available at: https://archive.ics.uci.edu/ml/datasets/FordA"""
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


class DatasetBuilder:
    """DatasetBuilder for FordDataset.

    Args:
        split (str): The split of the dataset. Either "train" or "test".
        use_k_fold (bool): Whether to use k-fold cross-validation.
        num_folds (None | int): Number of folds to use for k-fold cross
    """
    def __init__(self, split: str = "train", use_k_fold: bool = False, num_folds: None | int = None) -> None:
        self.root_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"
        self.split = split
        self.use_k_fold = use_k_fold
        self.shape =

        if self.split == "train":
            self.raw_data = torch.tensor(np.loadtxt(self.root_url + "FordA_TRAIN.tsv", delimiter="\t"), dtype=torch.float32)
        else:
            self.raw_data = torch.tensor(np.loadtxt(self.root_url + "FordA_TEST.tsv", delimiter="\t"), dtype=torch.float32)

        if self.use_k_fold:
            self.num_folds = num_folds
            self.skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

        self.labels = self.raw_data[:, 0]  # get first element from each example
        self.sequences = self.raw_data[:, 1:]  # get all elements after first element
        self.labels[self.labels == -1] = 0  # change all -1 labels to 0

    def get_dataset(self) -> FordDataset | dict[str, list[FordDataset]]:
        """Get dataset for the specified split.

        Returns: FordDataset object or a dictionary of 'num_folds' training and validation FordDataset objects when
        use_k_fold is True.
        """
        if self.use_k_fold:
            train_datasets = []
            val_datasets = []
            for train_idx, val_idx in self.skf.split(self.sequences, self.labels):
                train_datasets.append(FordDataset(self.sequences[train_idx], self.labels[train_idx]))
                val_datasets.append(FordDataset(self.sequences[val_idx], self.labels[val_idx]))
            return {"train": train_datasets, "val": val_datasets}

        return FordDataset(self.sequences, self.labels)

    def create_dataloader(self, batch_size: int = 16, shuffle: bool = False, drop_last: bool = False):
        """Create DataLoader/s for the specified split.

        Returns: DataLoader object or a tuple containing a training and validation object
        that are each a list of 'num_folds' DataLoader objects when use_k_fold is True.
        """
        dataset = self.get_dataset()
        if self.use_k_fold:
            train_dataloaders = []
            val_dataloaders = []
            for train_dataset, val_dataset in zip(dataset["train"], dataset["val"]):
                train_dataloaders.append(DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle))
                val_dataloaders.append(DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle))
            return train_dataloaders, val_dataloaders

        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

from src.datasets import FordDataset

from torch.utils.data import DataLoader

def get_dataloader(split, batch_size):
    dataset = FordDataset(split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

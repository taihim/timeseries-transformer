from src.timeseries_transformer.datasets import DatasetBuilder
from torch.utils.data import DataLoader

def get_dataloader(split, batch_size):
    dataset = DatasetBuilder(split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

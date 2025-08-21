import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset, ConcatDataset

if __name__ == '__main__':
    datasets = []

    for i in range(5):
        features = torch.rand(128, 3)
        labels = torch.randint(0, 2, (128,))

        datasets.append(TensorDataset(*[features, labels]))


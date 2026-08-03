import torch
from torch.utils.data import Dataset, ConcatDataset
import numpy as np

class EndoDataset(Dataset):
    def __init__(self, features, labels, window=64):
        self.x = torch.from_numpy(features).float()
        self.y = torch.tensor(labels, dtype=torch.long)
        self.w = window

    def __len__(self):
        return len(self.x) - self.w + 1

    def __getitem__(self, idx):
        x = self.x[idx:idx + self.w]
        y = self.y[idx + self.w - 1]
        return x, y

def build_dataset(features: list[np.ndarray], labels: list[list[int]], idx: list[int], window: int = 64):
    return ConcatDataset([
        EndoDataset(features[i], labels[i], window) for i in idx
    ])
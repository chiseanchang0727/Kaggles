import torch
import torch.nn as nn
from torch.utils.data import Dataset



class InputDataset(Dataset):

    def __init__(self, X, y):
        super().__init__()

        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)
    

    def __getitem__(self, index):
        return self.X[index], self.y[index]
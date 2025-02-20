import torch
import torch.nn as nn
from torch.utils.data import Dataset



class InputDataset(Dataset):

    def __init__(self, X, y):
        super().__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.X = torch.tensor(X.values, dtype=torch.float32).to(device)
        self.y = torch.tensor(y.values, dtype=torch.float32).to(device).view(-1, 1)

    def __len__(self):
        return len(self.X)
    

    def __getitem__(self, index):
        return self.X[index], self.y[index]
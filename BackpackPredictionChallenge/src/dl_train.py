import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.dataset import InputDataset
from src.utils import split_data


# def get_dataset

def get_dataset(df, batch_size):
    X_train, X_valid, y_train, y_valid = split_data(df)

    train_dataset = InputDataset(X_train, y_train)
    valid_dataset = InputDataset(X_valid, y_valid)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size, shuffle=True)

    return train_loader, valid_loader

# def get_model





def dl_train(df, batch_size=1000):

    train_loader, valid_loader = get_dataset(df, batch_size)
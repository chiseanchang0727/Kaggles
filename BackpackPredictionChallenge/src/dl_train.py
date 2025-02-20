import torch
import torch.nn as nn
from src.utils import split_data
from torch.utils.data import DataLoader
from src.dataset import InputDataset
from src.dl_model import NeuralNetwork


# def get_dataset

def get_dataset(df, batch_size):
    X_train, X_valid, y_train, y_valid = split_data(df)

    train_dataset = InputDataset(X_train, y_train)
    valid_dataset = InputDataset(X_valid, y_valid)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size, shuffle=True)

    return train_loader, valid_loader


def dl_train(df, save_model, model_w_dir, batch_size=1000, epochs=10, lr=0.001):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader, valid_loader = get_dataset(df, batch_size)

    input_dim = next(iter(train_loader))[0].shape[1]

    model = NeuralNetwork(input_dim)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for i, (X_batch, y_batch) in enumerate(train_loader):

            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Reset gradients to prevent accumulation from previous batches
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        valid_loss = 0.0

        with torch.no_grad():
            for X_val, y_val in valid_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                val_pred = model(X_val)
                val_loss = criterion(val_pred, y_val)
                valid_loss += val_loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Valid Loss: {valid_loss/len(valid_loader):.4f}")

        if save_model:
            torch.save(model.state_dict(), model_w_dir/'nn_model.pth')

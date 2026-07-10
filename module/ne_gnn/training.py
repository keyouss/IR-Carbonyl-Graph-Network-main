"""Training and evaluation helpers for NE-GNN."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch_geometric.loader import DataLoader


def train_one_epoch(model, loader: DataLoader, optimizer, criterion, device: str):
    model.train()
    losses = []
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch)
        loss = criterion(pred, batch.y.view(-1))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def predict(model, dataset, device: str = "cpu", batch_size: int = 128):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    y_true = []
    y_pred = []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch)
        y_true.append(batch.y.detach().cpu().reshape(-1))
        y_pred.append(pred.detach().cpu().reshape(-1))
    return torch.cat(y_true).numpy(), torch.cat(y_pred).numpy()


def evaluate_predictions(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_model(
    model,
    train_dataset,
    valid_dataset=None,
    epochs: int = 300,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: str = "cpu",
):
    """Train the model and return a history list."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        record = {"epoch": epoch, "train_loss": train_loss}
        if valid_dataset is not None:
            y_true, y_pred = predict(model, valid_dataset, device=device, batch_size=batch_size)
            record.update({f"valid_{k}": v for k, v in evaluate_predictions(y_true, y_pred).items()})
        history.append(record)
    return history

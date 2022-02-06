import torch
import numpy as np
from torch_geometric.loader import DataLoader

import sklearn.metrics as metrics
from sklearn.metrics import f1_score, precision_score, recall_score


def load_data(dataset, mol_num):
    assert mol_num < len(dataset)
    train_dataset = dataset[:150]
    test_dataset = dataset[150:]
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_loader, test_loader


def train(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
          criterion: torch.nn.Module, data_loader: DataLoader):
    model.train()
    for data in data_loader:  # Iterate in batches over the training dataset.
        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.


def test_old(model: torch.nn.Module, data_loader: DataLoader):
    model.eval()
    correct = 0
    for data in data_loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(data_loader.dataset)  # Derive ratio of correct predictions.


def test(model: torch.nn.Module, data_loader: DataLoader):
    with torch.no_grad():
        model.eval()
        GT = np.zeros(len(data_loader))
        preds = np.zeros(len(data_loader))
        probas = np.zeros(len(data_loader))

        i = 0
        for data in data_loader:
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1).item()
            GT[i] = data.y.item()
            preds[i] = pred

            probas[i] = out.softmax(dim=1).squeeze()[1].detach().clone().cpu().numpy()

            i += 1

        f1 = f1_score(GT, preds)
        precision = precision_score(GT, preds)
        recall = recall_score(GT, preds)
        auprc = metrics.average_precision_score(GT, probas)
        auroc = metrics.roc_auc_score(GT, probas)

        return f1, precision, recall, auprc, auroc
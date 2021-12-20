import torch
from torch_geometric.data import DataLoader


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


def test(model: torch.nn.Module, data_loader: DataLoader):
     model.eval()
     correct = 0
     for data in data_loader:  # Iterate in batches over the training/test dataset.
         out = model(data.x, data.edge_index, data.batch)
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
     return correct / len(data_loader.dataset)  # Derive ratio of correct predictions.

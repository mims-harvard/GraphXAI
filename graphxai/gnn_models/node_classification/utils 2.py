import torch
from torch_geometric.data import Data


def train(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
          criterion: torch.nn.Module, data: Data, losses: list = None):
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(data.x, data.edge_index)  # Perform a single forward pass.
    # print('Out shape', out.shape)
    # print('y shape', data.y.shape)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.

    if losses is not None:
        losses.append(loss.item())

    return loss


def test(model: torch.nn.Module, data: Data, test_accs: list = None):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.

    if test_accs is not None:
        test_accs.append(test_acc)

    return test_acc

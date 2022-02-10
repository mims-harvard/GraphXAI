import os, sys; sys.path.append('..')
import torch
from torch_geometric.datasets import TUDataset

from graphxai.explainers import IntegratedGradExplainer
#from graphxai.explainers.utils.visualizations import visualize_mol_explanation
from graphxai.gnn_models.graph_classification import train, test
from graphxai.gnn_models.graph_classification.gcn import GCN_2layer, GCN_3layer
from graphxai.gnn_models.graph_classification.gin import GIN_2layer, GIN_3layer
from graphxai.datasets import Benzene

from utils import get_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load data: ------------------------------------------
#dataset = Benzene(split_sizes = (0.8, 0.2, 0), seed = seed)
dataset = get_dataset('benzene', device = device)
train_loader, _ = dataset.get_train_loader(batch_size = 64)
test_loader, _ = dataset.get_test_loader()
val_loader, _ = dataset.get_val_loader()

# Train GNN model -------------------------------------
model = GIN_3layer(14, 32, 2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss().to(device)

best_f1 = 0
for epoch in range(1, 101):
    train(model, optimizer, criterion, train_loader)
    #f1, prec, rec, auprc, auroc = test(model, test_loader)
    f1, prec, rec, auprc, auroc = test(model, val_loader)

    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), os.path.join('model_weights', f'GIN_fc.pth'))

    print(f'Epoch: {epoch:03d}, Test F1: {f1:.4f}, Test AUROC: {auroc:.4f}')

f1, precision, recall, auprc, auroc = test(model, test_loader)
print(f'Test F1: {f1:.4f}, Test AUROC: {auroc:.4f}')
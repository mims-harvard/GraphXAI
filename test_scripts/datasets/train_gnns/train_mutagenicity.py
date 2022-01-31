import sys
import torch
from torch_geometric.datasets import TUDataset

from graphxai.explainers import PGMExplainer
#from graphxai.explainers.utils.visualizations import visualize_mol_explanation
from graphxai.gnn_models.graph_classification import train, test
from graphxai.gnn_models.graph_classification.gcn import GCN_2layer, GCN_3layer
from graphxai.gnn_models.graph_classification.gin import GIN_2layer, GIN_3layer
from graphxai.datasets import Mutagenicity

if len(sys.argv) > 1:
    seed = int(sys.argv[1])
else:
    seed = 1200

# Load data: ------------------------------------------
dataset = Mutagenicity(root = './data/', split_sizes = (0.8, 0.2, 0), seed = seed)
train_loader, _ = dataset.get_train_loader(batch_size = 64)
test_loader, _ = dataset.get_test_loader()

# Train GNN model -------------------------------------
model = GIN_3layer(14, 32, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(1, 31):
    train(model, optimizer, criterion, train_loader)
    f1, prec, rec, auprc, auroc = test(model, test_loader)
    print(f'Epoch: {epoch:03d}, Test F1: {f1:.4f}, Test AUROC: {auroc:.4f}')
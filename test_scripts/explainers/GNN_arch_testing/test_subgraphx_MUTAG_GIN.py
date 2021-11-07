import sys
import numpy as np

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader, Data
from torch_geometric.nn import global_mean_pool, GCNConv
import torch.nn.functional as F

#from explainers.guidedbp import GuidedBP
from graphxai.gnn_models.graph_classification import GIN, load_data, train, test
from graphxai.explainers.utils.visualizations import *

import matplotlib.pyplot as plt

dataset = TUDataset(root='data/TUDataset', name='MUTAG')

assert len(sys.argv) == 2, "usage: python3 test_GBP_MUTAG.py <molecular index>"
mol_num = int(sys.argv[1])
assert mol_num < len(dataset)

train_loader, test_loader = load_data(dataset, mol_num)

model = GIN(hidden_channels=64, in_channels = dataset.num_node_features, out_channels = dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()


for epoch in range(1, 201):
    train(model, optimizer, criterion, train_loader)
    train_acc = test(model, train_loader)
    test_acc = test(model, test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')


mol = dataset[mol_num]

atom_list = ['C', 'N', 'O', 'F', 'I', 'Cl', 'Br']
atom_map = {i:atom_list[i] for i in range(len(atom_list))}

atoms = []
for i in range(mol.x.shape[0]):
    atoms.append(atom_map[mol.x[i,:].tolist().index(1)])

model.eval()
pred = model(mol.x, mol.edge_index, torch.zeros(1).type(torch.int64))
pred_class = pred.argmax(dim=1).item()

from graphxai.explainers.subgraphx import SubgraphX

explainer = SubgraphX(model, reward_method = 'gnn_score')

exp = explainer.get_explanation_graph(
    mol.x, 
    edge_index = mol.edge_index,  
    max_nodes = 10,
    forward_kwargs = {'batch':torch.zeros(1, dtype = torch.long)} 
)

fig, (ax1, ax2) = plt.subplots(1, 2)

# Visualize molecule:
visualize_mol_explanation(mol, atoms = atoms, ax = ax1, show = False)
ax1.set_title('Original Molecule')

visualize_mol_explanation(
    mol, 
    node_weights = exp['feature'].type(torch.int32).tolist(), 
    atoms = atoms, 
    ax = ax2, 
    show = False, 
    directed = False)
ax2.set_title('Subgraph Explanation')

ymin, ymax = ax1.get_ylim()
xmin, xmax = ax1.get_xlim()
ax1.text(xmin, ymax - 0.1*(ymax-ymin), 'Label = {:d}'.format(mol.y.item()))
ax1.text(xmin, ymax - 0.15*(ymax-ymin), 'Pred  = {:d}'.format(pred.argmax(dim=1).item()))
plt.show()

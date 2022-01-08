import sys
import numpy as np
import torch
from torch_geometric.datasets import TUDataset

from graphxai.explainers import IntegratedGradExplainer
from graphxai.explainers.utils.visualizations import visualize_mol_explanation
from graphxai.gnn_models.graph_classification import GCN, load_data, train, test

import matplotlib.pyplot as plt


dataset = TUDataset(root='data/TUDataset', name='MUTAG')
assert len(sys.argv) == 2, "Must provide index (0 - {}) for molucule in dataset.".format(len(dataset))
mol_num = int(sys.argv[1])
train_loader, test_loader = load_data(dataset, mol_num)
model = GCN(in_channels=dataset.num_node_features, hidden_channels=64,
            out_channels=dataset.num_classes)
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

explainer = IntegratedGradExplainer(model, criterion)

model.eval()
pred = model(mol.x, mol.edge_index, torch.zeros(1).type(torch.int64))

print('GROUND TRUTH LABEL: \t {}'.format(mol.y.item()))
print('PREDICTED LABEL   : \t {}'.format(pred.argmax(dim=1).item()))

exp = explainer.get_explanation_graph(mol.edge_index, mol.x, mol.y,
                                      forward_args = (torch.zeros(1).type(torch.int64),))

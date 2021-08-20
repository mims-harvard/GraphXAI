import sys

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import remove_self_loops

from graphxai.explainers import GNN_LRP
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

pred = model(mol.x, mol.edge_index, torch.zeros(1).type(torch.int64))
pred_class = pred.argmax(dim=1).item()

print('GROUND TRUTH LABEL: \t {}'.format(mol.y.item()))
print('PREDICTED LABEL   : \t {}'.format(pred.argmax(dim=1).item()))

from graphxai.explainers.gnn_lrp import GNN_LRP

gnn_lrp = GNN_LRP(model)
exp = gnn_lrp.get_explanation_graph(
    mol.x, edge_index = mol.edge_index, label = pred_class, 
    forward_kwargs = {'batch':torch.tensor([1], dtype = torch.long)}
)

# For purposes of visualization, remove self-loops from edge index and explanations:
mol.edge_index, edge_exp_label = remove_self_loops(edge_index=exp.graph.edge_index, edge_attr = exp.edge_imp)

fig, ax = plt.subplots()

visualize_mol_explanation(mol, atoms = atoms, edge_weights = edge_exp_label.tolist(), 
    weight_map = True, directed = True, ax = ax, show = False, fig = fig)

ymin, ymax = ax.get_ylim()
xmin, xmax = ax.get_xlim()
ax.text(xmin, ymax - 0.1*(ymax-ymin), 'Label = {:d}'.format(mol.y.item()))
ax.text(xmin, ymax - 0.15*(ymax-ymin), 'Pred  = {:d}'.format(pred_class))

fig.suptitle('GNN-LRP')
plt.show()

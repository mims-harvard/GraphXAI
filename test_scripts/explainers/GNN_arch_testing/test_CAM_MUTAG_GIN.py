import sys

import torch
from torch_geometric.datasets import TUDataset

from graphxai.explainers import CAM, GradCAM
from graphxai.explainers.utils.visualizations import visualize_mol_explanation
from graphxai.gnn_models.graph_classification import GIN, load_data, train, test

import matplotlib.pyplot as plt


dataset = TUDataset(root='data/TUDataset', name='MUTAG')
assert len(sys.argv) == 2, "Must provide index (0 - {}) for molucule in dataset.".format(len(dataset))
mol_num = int(sys.argv[1])
train_loader, test_loader = load_data(dataset, mol_num)
model = GIN(in_channels=dataset.num_node_features, hidden_channels=64,
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

cam = GradCAM(model, criterion = criterion)

model.eval()
pred = model(mol.x, mol.edge_index, torch.zeros(1).type(torch.int64))

print('GROUND TRUTH LABEL: \t {}'.format(mol.y.item()))
print('PREDICTED LABEL   : \t {}'.format(pred.argmax(dim=1).item()))

exp = cam.get_explanation_graph(mol.x, edge_index = mol.edge_index, label = mol.y,
            forward_kwargs = {'batch':torch.zeros(1).type(torch.int64)})

# Show the plots of both CAM and Grad-CAM on separate axes:
fig, (ax1, ax2) = plt.subplots(1, 2)

visualize_mol_explanation(mol, exp['feature'], atoms = atoms, ax = ax1, show = False)

# Get CAM explanation:
act = lambda x: torch.argmax(x, dim=1)
cam = CAM(model, activation = act)

exp = cam.get_explanation_graph(mol.x, edge_index = mol.edge_index, label = mol.y,
            forward_kwargs = {'batch':torch.zeros(1).type(torch.int64)})

visualize_mol_explanation(mol, exp['feature'], atoms = atoms, ax = ax2, show = False, fig = fig)

ax1.set_title('Grad-CAM')
ax2.set_title('CAM')

ymin, ymax = ax1.get_ylim()
xmin, xmax = ax1.get_xlim()
ax1.text(xmin, ymax - 0.1*(ymax-ymin), 'Label = {:d}'.format(mol.y.item()))
ax1.text(xmin, ymax - 0.15*(ymax-ymin), 'Pred  = {:d}'.format(pred.argmax(dim=1).item()))

plt.show()

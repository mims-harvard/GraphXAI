import sys
sys.path.append('..')
import numpy as np

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool, GCNConv
import torch.nn.functional as F

from gnn_ex_eval.explainers.cam import CAM, Grad_CAM
from gnn_ex_eval.models.gcn import GCN_graph
from gnn_ex_eval.explainers.utils.visualizations import visualize_mol_explanation

import matplotlib.pyplot as plt

dataset = TUDataset(root='data/TUDataset', name='MUTAG')

#dataset = dataset.shuffle()
train_dataset = dataset[:150]
test_dataset = dataset[150:]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#model = GCN_graph(nfeat=7, nhid=64, nclass=2)

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        #torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x

# model = GNN(hidden_channels=64)

# #optimizer = torch.optim.Adam(model.parameters(), lr=1)
# optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
# criterion = torch.nn.CrossEntropyLoss()

model = GCN(hidden_channels=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
         out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
         loss = criterion(out, data.y)  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.

def test(loader):
     model.eval()

     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         out = model(data.x, data.edge_index, data.batch)  
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
     return correct / len(loader.dataset)  # Derive ratio of correct predictions.


for epoch in range(1, 201):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')


mol = dataset[98]

atom_list = ['C', 'N', 'O', 'F', 'I', 'Cl', 'Br']
atom_map = {i:atom_list[i] for i in range(len(atom_list))}

atoms = []
for i in range(mol.x.shape[0]):
    atoms.append(atom_map[mol.x[i,:].tolist().index(1)])

cam = Grad_CAM(model, '', criterion = criterion)

model.eval()
pred = model(mol.x, mol.edge_index, torch.zeros(1).type(torch.int64))

print('GROUND TRUTH LABEL: \t {}'.format(mol.y.item()))
print('PREDICTED LABEL   : \t {}'.format(pred.argmax(dim=1).item()))

exp = cam.get_explanation_graph(mol.x, mol.y, mol.edge_index,
            forward_args = (torch.zeros(1).type(torch.int64),))

# exp = np.interp(exp, (np.min(exp), np.max(exp)), (0, 10))
# print(exp)

fig, (ax1, ax2) = plt.subplots(1, 2)


visualize_mol_explanation(mol, exp, atoms = atoms, ax = ax1, show = False)

act = lambda x: torch.argmax(x, dim=1)
cam = CAM(model, '', activation = act)

exp = cam.get_explanation_graph(mol.x, mol.edge_index,
            forward_args = (torch.zeros(1).type(torch.int64),))

visualize_mol_explanation(mol, exp, atoms = atoms, ax = ax2, show = False)


ax1.set_title('Grad-CAM')
ax2.set_title('CAM')


ymin, ymax = ax1.get_ylim()
xmin, xmax = ax1.get_xlim()
ax1.text(xmin, ymax - 0.1*(ymax-ymin), 'Label = {:d}'.format(mol.y.item()))
ax1.text(xmin, ymax - 0.15*(ymax-ymin), 'Pred  = {:d}'.format(pred.argmax(dim=1).item()))

# ax1.colorbar()
# ax2.colorbar()

plt.show()
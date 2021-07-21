import sys
sys.path.append('..')
import numpy as np

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool, GCNConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch.nn.functional as F

from explainers.guidedbp import GuidedBP
from gnn_ex_eval.models.gcn import GCN_graph
from explainers.utils.visualizations import visualize_mol_explanation, parse_GNNLRP_explanations

import matplotlib.pyplot as plt
from torch_sparse import SparseTensor
from torch import Tensor

dataset = TUDataset(root='data/TUDataset', name='MUTAG')

#mol_num = int(sys.argv[1])
#assert mol_num < len(dataset)

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


for epoch in range(1, 101):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')


mol = dataset[161]

atom_list = ['C', 'N', 'O', 'F', 'I', 'Cl', 'Br']
atom_map = {i:atom_list[i] for i in range(len(atom_list))}

atoms = []
for i in range(mol.x.shape[0]):
    atoms.append(atom_map[mol.x[i,:].tolist().index(1)])

pred = model(mol.x, mol.edge_index, torch.zeros(1).type(torch.int64))
pred_class = pred.argmax(dim=1).item()

print('GROUND TRUTH LABEL: \t {}'.format(mol.y.item()))
print('PREDICTED LABEL   : \t {}'.format(pred.argmax(dim=1).item()))

#from dig.xgraph.method import GNN_GI, GNN_LRP
from explainers.gnn_lrp import GNN_LRP

print('Mol edge index shape', mol.edge_index.shape)

gnn_lrp = GNN_LRP(model, explain_graph = True)
walks, edge_masks, related_predictions = gnn_lrp.forward(
    mol.x, mol.edge_index, num_classes = 2,
    forward_args = (torch.tensor([1], dtype = torch.long),)
)

# print('Walks:', walks)
# print('Edge masks:', edge_masks)
# print('Related Predictions:', related_predictions)

# print('Walk ids:', walks['ids'].unique().tolist())

node_exp, edge_exp = parse_GNNLRP_explanations((walks, edge_masks, related_predictions), mol.edge_index, pred_class)

print('Len edge explanations', len(edge_exp))

mol.edge_attr = edge_exp

visualize_mol_explanation(mol, atoms = atoms, edge_weights = edge_exp, weight_map = True, directed = True)
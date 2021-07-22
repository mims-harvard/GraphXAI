import sys, random
sys.path.append('..')
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from graphxai.explainers.utils.testing_datasets import BA_houses_maker as BAH
from graphxai.explainers.utils.visualizations import *

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import k_hop_subgraph, subgraph, to_networkx
from torch_geometric.data import Data

def get_data(n, m, num_houses):
    bah = BAH(n, m)
    BAG = bah.make_BA_shapes(num_houses)
    data = bah.make_data(BAG)
    inhouse = bah.in_house
    return data, list(inhouse)

n = 500
m = 2
num_houses = 20

data, inhouse = get_data(n, m, num_houses)


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, input_feat, classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_feat, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

model = GCN(64, input_feat = 1, classes = 2)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

def train():
      model.train()
      optimizer.zero_grad()  # Clear gradients.
      out = model(data.x, data.edge_index)  # Perform a single forward pass.
    #   print('Out shape', out.shape)
    #   print('y shape', data.y.shape)
      loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
      loss.backward()  # Derive gradients.
      optimizer.step()  # Update parameters based on gradients.
      return loss

def test():
      model.eval()
      out = model(data.x, data.edge_index)
      pred = out.argmax(dim=1)  # Use the class with highest probability.
      test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
      return test_acc

for epoch in range(1, 201):
    loss = train()
    acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Acc: {acc:.4f}')


node_idx = int(random.choice(inhouse))

model.eval()
pred = model(data.x, data.edge_index)[node_idx, :].reshape(-1, 1)
pred_class = pred.argmax(dim=0).item()

from graphxai.explainers.gnn_lrp import GNN_LRP

gnn_lrp = GNN_LRP(model, explain_graph=False)
# walks, edge_masks, related_predictions, khop_info = gnn_lrp.forward(
#     data.x, data.edge_index, num_classes = 2,
#     node_idx = node_idx
# )
# edge_scores, khop_info = gnn_lrp.forward(
#     data.x, data.edge_index, num_classes = 2,
#     node_idx = node_idx
# )
edge_scores, khop_info = gnn_lrp.get_explanation_node(
    x = data.x, 
    node_idx = node_idx,
    edge_index = data.edge_index, 
    num_classes = 2
)

print(edge_scores)
print(khop_info)

visualize_subgraph_explanation_w_edge(khop_info[0], khop_info[1], edge_weights = edge_scores[pred_class],
    node_idx=node_idx)

# node_exp, edge_exp = parse_GNNLRP_explanations((walks, edge_masks, related_predictions), data.edge_index, pred_class)

# # 2 = # GCN layers
# #subgraph_nodes, subgraph_eidx, _, edge_mask = k_hop_subgraph(node_idx, 2, data.edge_index)
# #subgraph_eidx, subgraph_exp = subgraph(subgraph_nodes, edge_index=data.edge_index, edge_attr=torch.tensor(edge_exp))


# print('Walks', walks)
# print('Edge masks', edge_masks)
# print('Related predictions', related_predictions)

#edge_inds = 
# edges = (edge_masks[pred_class] > 0).nonzero(as_tuple=True)[0]
# subgraph_edge_exp = [edge_exp[e] for e in edges]


# print('Edge mask stats:')
# for e in edge_masks:
#     for i in e.unique():
#         print(i, (e > 0).nonzero(as_tuple=True)[0])
#     print('')

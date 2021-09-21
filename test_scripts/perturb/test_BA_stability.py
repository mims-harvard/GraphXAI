import random
import torch
import numpy as np

from torch_geometric.data import Data

from graphxai.utils.perturb import rewire_edges
from graphxai.gnn_models.node_classification import BA_Houses, GCN, train, test
from graphxai.explainers import GNNExplainer
from graphxai.utils.representation import extract_step


def rewire_data(data: Data, node_idx: int, k_hops: int = 3, rewire_prob: float = 0.01):
    new_edge_index = rewire_edges(data.edge_index, num_nodes=data.num_nodes,
                                  node_idx=node_idx, k_hops=k_hops,
                                  rewire_prob=rewire_prob)
    new_data = Data(edge_index=new_edge_index,
                    x=data.x, y=data.y, test_mask=data.test_mask)
    return new_data


n = 300
m = 2
num_houses = 20

bah = BA_Houses(n, m)
data, inhouse = bah.get_data(num_houses, multiple_features=True)

model = GCN(64, input_feat=3, classes=2)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

node_idx = int(random.choice(inhouse))

for epoch in range(1, 201):
    loss = train(model, optimizer, criterion, data)
    acc = test(model, data)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Acc: {acc:.4f}')

# Rewire edges of node_idx's enclosing subgraph
new_data = rewire_data(data, node_idx, k_hops=3, rewire_prob=0.01)
new_acc = test(model, new_data)

# Get walk steps
walk_steps, _ = extract_step(model, data.x, data.edge_index)
new_walk_steps, _ = extract_step(model, new_data.x, new_data.edge_index)

# Compute normalized L2 distance of representation
layers = [e['output'] for e in walk_steps]
layers = [e/torch.norm(e, dim=1).view(-1, 1) for e in layers]
new_layers = [e['output'] for e in new_walk_steps]
new_layers = [e/torch.norm(e, dim=1).view(-1, 1) for e in new_layers]
diffs = [torch.norm(new-old, dim=1) for old, new in zip(layers, new_layers)]

print(diffs[0].mean().item())
print(diffs[1].mean().item())

# Test the explanation change
explainer = GNNExplainer(model)

def get_exp(node_idx, data):
    exp, _ = explainer.get_explanation_node(node_idx, data.x, data.edge_index,
                                            label=data.y, explain_feature=False)
    return exp['edge_imp']

exp = get_exp(node_idx, data)
new_exp = get_exp(node_idx, new_data)

import random
import torch

from torch_geometric.data import Data
from torch_geometric.utils import degree
from torch_geometric.utils.subgraph import subgraph
from torch_geometric.utils.random import erdos_renyi_graph
from torch_geometric.utils.convert import to_scipy_sparse_matrix

from IPython.display import SVG
from sknetwork.visualization import svg_graph

from old import sbm_with_singletons
from graphxai.gnn_models.node_classification import GCN, train, test
from graphxai.utils.representation import extract_step
from graphxai.explainers import GNNExplainer


block_sizes = [40, 45, 50]
num_singletons = 10
p_out = 0.02

num_community_nodes = sum(block_sizes)
num_nodes = num_community_nodes + num_singletons
label = torch.ones(num_nodes, dtype=int)
label[-num_singletons:] = 0

# Create train / test mask
test_size = 0.3
train_mask = torch.full((num_nodes,), False)
test_mask = torch.full((num_nodes,), False)
test_set = set(random.sample(list(range(num_nodes)), int(test_size * num_nodes)))
for i in range(num_nodes):
    if i in test_set:
        test_mask[i] = True
    else:
        train_mask[i] = True

# Sample a SBM
edge_index = sbm_with_singletons(block_sizes, num_singletons,
                                 p_in=[0.5, 0.6, 0.7], p_out=p_out)

# Use degree as feature in order to achieve 100% accuracy
# x = degree(edge_index[0], num_nodes).view(-1, 1)
x = torch.randn(num_nodes, 1)
data = Data(x=x, edge_index=edge_index, y=label,
            train_mask=train_mask, test_mask=test_mask)

model = GCN(32, input_feat=1, classes=2)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

acc = 0
for epoch in range(1, 201):
    loss = train(model, optimizer, criterion, data)
    acc = test(model, data)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Acc: {acc:.4f}')

# Rewire edges incident to some "isolated" nodes
def rewire(num_nodes_to_rewire):
    num_nodes_to_keep = num_nodes - num_nodes_to_rewire
    kept_edge_index, _ = subgraph(torch.arange(num_nodes_to_keep),
                                  edge_index, num_nodes=num_nodes)
    added_edge_index = erdos_renyi_graph(num_nodes, edge_prob=p_out)
    added_edge_index, _ = subgraph(torch.arange(num_nodes_to_keep, num_nodes),
                                   added_edge_index, num_nodes=num_nodes)
    new_edge_index = torch.cat([kept_edge_index, added_edge_index], dim=1)
    new_data = Data(x=x, edge_index=new_edge_index, y=label,
                    train_mask=train_mask, test_mask=test_mask)
    return new_data

new_data = rewire(num_nodes_to_rewire=1)
new_acc = test(model, new_data)
print(acc)
print(new_acc)

# Visualize the old and new graphs
adj = to_scipy_sparse_matrix(edge_index).tocsr()
img = svg_graph(adj)
new_adj = to_scipy_sparse_matrix(new_edge_index).tocsr()
new_img = svg_graph(new_adj)
# Run in notebook to see visualization
SVG(img)
SVG(new_img)

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

def get_exp(data):
    exp, _ = explainer.get_explanation_node(node_idx, data.x, data.edge_index,
                                            label=data.y, explain_feature=False)
    return exp['edge_imp']

exp = get_exp(data)
new_exp = get_exp(new_data)

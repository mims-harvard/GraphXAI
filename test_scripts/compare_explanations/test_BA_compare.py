"""
(stability)
1) edge mask of original graph by some explainer
2) edge mask of perturbed graph by some explainer

(faithfulness)
1) edge mask of original graph by some explainer
2) ground truth edge mask
Compute importance score
"""

import random
import torch

from torch_geometric.data import Data

from graphxai.utils.perturb import rewire_edges
from graphxai.gnn_models.node_classification import BA_Houses, GCN, train, test
from graphxai.explainers import GNNExplainer, PGExplainer
# from graphxai.utils.representation import extract_step
from graphxai.visualization import visualize_edge_explanation

# Set random seeds
seed = 1
torch.manual_seed(seed)
random.seed(seed)

def rewire_data(data: Data, node_idx: int, num_hops: int = 3, rewire_prob: float = 0.01):
    new_edge_index = rewire_edges(data.edge_index, num_nodes=data.num_nodes,
                                  node_idx=node_idx, num_hops=num_hops,
                                  rewire_prob=rewire_prob)
    new_data = Data(edge_index=new_edge_index,
                    x=data.x, y=data.y, test_mask=data.test_mask)
    return new_data


n = 300
m = 2
num_houses = 20

bah = BA_Houses(n, m)
data, inhouse = bah.get_data(num_houses, null_feature=True)

model = GCN(8, input_feat=2, classes=2)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

node_idx = int(random.choice(inhouse))

losses = []
test_accs = []
for epoch in range(1, 201):
    loss = train(model, optimizer, criterion, data, losses)
    acc = test(model, data, test_accs)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Acc: {acc:.4f}')

# # Rewire edges of node_idx's enclosing subgraph
# new_data = rewire_data(data, node_idx, num_hops=3, rewire_prob=0.01)
# new_acc = test(model, new_data)

# # This part compares the internal representation
# # Get walk steps
# walk_steps, _ = extract_step(model, data.x, data.edge_index)
# new_walk_steps, _ = extract_step(model, new_data.x, new_data.edge_index)

# # Compute normalized L2 distance of representation
# layers = [e['output'] for e in walk_steps]
# layers = [e/torch.norm(e, dim=1).view(-1, 1) for e in layers]
# new_layers = [e['output'] for e in new_walk_steps]
# new_layers = [e/torch.norm(e, dim=1).view(-1, 1) for e in new_layers]
# diffs = [torch.norm(new-old, dim=1) for old, new in zip(layers, new_layers)]

# print(diffs[0].mean().item())
# print(diffs[1].mean().item())

# Test edge explainers

def get_exp(explainer, node_idx, data):
    exp, khop_info = explainer.get_explanation_node(
        node_idx, data.x, data.edge_index, label=data.y, num_hops=2)
    return exp['edge_imp'], khop_info[0], khop_info[1]

# GNNExplainer
# correctnesses = []
# for i in range(10):
#     node_idx = int(inhouse[i])
gnnexpr = GNNExplainer(model, seed=seed)
edge_imp, subset, sub_edge_index = get_exp(gnnexpr, node_idx, data)
# new_edge_imp, new_subset, new_sub_edge_index = get_exp(gnnexpr, node_idx, new_data)
sub_node_idx = -1
for sub_idx, idx in enumerate(subset.tolist()):
    if idx == node_idx:
        sub_node_idx = sub_idx
visualize_edge_explanation(sub_edge_index, num_nodes=len(subset),
                           node_idx=sub_node_idx, edge_imp=edge_imp)
# visualize_edge_explanation(new_sub_edge_index, num_nodes=new_subset.shape[0], edge_imp=new_edge_imp)

# Compare with ground truth unique explanation
# Locate which house
true_nodes, true_edges = [(nodes, edges) for nodes, edges in bah.houses if node_idx in nodes][0]
TPs = []
FPs = []
FNs = []
for i, edge in enumerate(sub_edge_index.T):
    # Restore original node numbering
    edge_ori = tuple(subset[edge].tolist())
    positive = edge_imp[i].item() > 0.8
    if positive:
        if edge_ori in true_edges:
            TPs.append(edge_ori)
        else:
            FPs.append(edge_ori)
    else:
        if edge_ori in true_edges:
            FNs.append(edge_ori)
TP = len(TPs)
FP = len(FPs)
FN = len(FNs)
correctness = TP / (TP + FP + FN)
# correctnesses.append(correctness)
print(f'Correctness score of gnn explainer is {correctness}')

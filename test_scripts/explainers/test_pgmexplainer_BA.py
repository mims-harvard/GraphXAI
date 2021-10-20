import random
import torch

from graphxai.explainers import PGMExplainer
from graphxai.datasets.feature import make_network_stats_feature
from graphxai.gnn_models.node_classification import BA_Houses, GCN, train, test


# Set random seeds
seed = 0
torch.manual_seed(seed)
random.seed(seed)

n = 300
m = 2
num_houses = 30

bah = BA_Houses(n, m)
data, inhouse = bah.get_data(num_houses)
data.x, feature_mask, feature_names = make_network_stats_feature(data.edge_index)

model = GCN(16, input_feat=data.x.shape[1], classes=2)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(1, 201):
    loss = train(model, optimizer, criterion, data)
    acc = test(model, data)

node_idx = random.choice(inhouse)

model.eval()
explainer = PGMExplainer(model, explain_graph=False)

node_idx = int(node_idx)
pgm, node_imp, khopinfo = explainer.get_explanation_node(node_idx, data.x,
                                                         data.edge_index, top_k_nodes=8)
exp_nodes = set(node_imp.nonzero().reshape(-1).tolist())
true_nodes, true_edges = [(nodes, edges) for nodes, edges in bah.houses
                          if node_idx in nodes][0]
precision = len(exp_nodes & true_nodes) / len(exp_nodes)
recall = len(exp_nodes & true_nodes) / len(true_nodes)
F = 1 / (1/precision + 1/recall)
print(f'F score: {F}')

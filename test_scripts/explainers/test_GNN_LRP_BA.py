import random
import torch
import numpy as np
import matplotlib.pyplot as plt

from graphxai.explainers import GNN_LRP
from graphxai.explainers.utils.visualizations import visualize_subgraph_explanation
from graphxai.gnn_models.node_classification import BA_Houses, GCN, train, test


n = 300
m = 2
num_houses = 20

bah = BA_Houses(n, m)
data, inhouse = bah.get_data(num_houses)

model = GCN(64, input_feat = 1, classes = 2)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(1, 201):
    loss = train(model, optimizer, criterion, data)
    acc = test(model, data)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Acc: {acc:.4f}')

node_idx = random.choice(inhouse)

model.eval()
pred = model(data.x, data.edge_index)[node_idx, :].reshape(-1, 1)
pred_class = pred.argmax(dim=0).item()

gnn_lrp = GNN_LRP(model, explain_graph=False)


exp, khop_info = gnn_lrp.get_explanation_node(
    x = data.x,
    label = int(pred_class),
    node_idx = int(node_idx),
    edge_index = data.edge_index,
    get_edge_scores=True,
    edge_aggregator=torch.sum
)

# visualize_subgraph_explanation_w_edge(khop_info[0], khop_info[1], edge_weights = exp['edge'][pred_class],
#     node_idx=node_idx, show = False)

visualize_subgraph_explanation(khop_info[1], edge_weights = exp['edge'], node_idx = int(node_idx),
show = False)

plt.title('GNN-LRP Explanation for Node {}'.format(node_idx))
plt.show()

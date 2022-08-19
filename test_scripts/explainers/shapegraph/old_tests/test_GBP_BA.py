import random
import torch
import matplotlib.pyplot as plt

from graphxai.explainers import GuidedBP
from graphxai.explainers.utils.visualizations import visualize_subgraph_explanation
from graphxai.gnn_models.node_classification import BA_Houses, train, test
from graphxai.gnn_models.node_classification.testing import GCN_3layer_basic
from graphxai.datasets import BAHouses, BAShapes
from graphxai.datasets.shape_graph import ShapeGraph

from graphxai.visualization.explanation_vis import visualize_node_explanation

# Set up dataset:
bah = ShapeGraph(model_layers = 3)
data = bah.get_graph(use_fixed_split=True)

# Get nodes in the house:
inhouse = (data.y == 1).nonzero(as_tuple=True)[0]

model = GCN_3layer_basic(64, input_feat = 10, classes = 2)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(1, 201):
    loss = train(model, optimizer, criterion, data)
    acc = test(model, data)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Acc: {acc:.4f}')

node_idx = random.choice(inhouse)

gbp = GuidedBP(model, criterion)
exp = gbp.get_explanation_node(data.x, data.y, edge_index = data.edge_index, node_idx = int(node_idx))

# Mask y according to subgraph:

fig, (ax1, ax2) = plt.subplots(1, 2)

ground_truth = bah.explanations[node_idx] # Get Explanation object
ground_truth.visualize_node(num_hops = bah.num_hops, graph_data = data, additional_hops = 1, heat_by_exp = True, ax=ax1)
ax1.set_title('Ground Truth')

# Aggregate node_imp for visualization:
exp.node_imp = [torch.sum(exp.node_imp[i]).item() for i in range(len(exp.node_imp))]
exp.visualize_node(num_hops = bah.num_hops, graph_data = data, additional_hops = 1, heat_by_exp = True, ax=ax2)
ax2.set_title('Guided Backprop (Explanation wrt Degree)')

model.eval()
pred = model(data.x, data.edge_index)[node_idx, :].reshape(-1, 1)

ymin, ymax = ax1.get_ylim()
xmin, xmax = ax1.get_xlim()
ax1.text(xmin, ymax - 0.1*(ymax-ymin), 'Label = {:d}'.format(data.y[node_idx].item()))
ax1.text(xmin, ymax - 0.15*(ymax-ymin), 'Pred  = {:d}'.format(pred.argmax(dim=0).item()))
plt.show()
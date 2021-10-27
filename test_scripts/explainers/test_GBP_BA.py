import random
import torch
import matplotlib.pyplot as plt

from graphxai.explainers import GuidedBP
from graphxai.explainers.utils.visualizations import visualize_subgraph_explanation
from graphxai.gnn_models.node_classification import BA_Houses, GCN, train, test
from graphxai.datasets import BAHouses, BAShapes

from graphxai.visualization.explanation_vis import visualize_node_explanation

n = 300
m = 2
num_houses = 20

hyp = {
    'num_hops': 1,
    'n': n,
    'm': m,
    'num_shapes': num_houses,
    'shape_insert_strategy': 'bound_12',
    'labeling_method': 'edge',
    'shape_upper_bound': 1,
    'feature_method': 'gaussian_lv'
}

# bah = BAShapes(
#     num_hops = 2,
#     n = 2000,
#     m = 1,
#     num_shapes = None,
#     shape_insert_strategy = 'neighborhood upper bound'
# )
bah = BAShapes(**hyp)
#data, inhouse = bah.get_data(num_houses, multiple_features=True)
data = bah.get_graph(use_fixed_split=True)

# Get nodes in the house:
inhouse = (data.y == 0).nonzero(as_tuple=True)[0]

model = GCN(64, input_feat = 10, classes = 2)
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

#visualize_node_explanation(ground_truth, ax = ax1, show = False)
ground_truth.context_draw(num_hops = bah.num_hops, additional_hops = 1, heat_by_exp = True, ax=ax1)
ax1.set_title('Ground Truth')

# Use first dimension of explanation matrix as the displayed explanation value:
# visualize_subgraph_explanation(exp.enc_subgraph.edge_index, 
#     [exp.node_imp[i,0].item() for i in range(exp.node_imp.shape[0])], 
#     node_idx = int(node_idx), ax = ax2, show = False)

# Aggregate node_imp for visualization:
exp.node_imp = [torch.sum(exp.node_imp[i]).item() for i in range(len(exp.node_imp))]

exp.context_draw(num_hops = bah.num_hops, additional_hops = 1, heat_by_exp = True, ax=ax2)
ax2.set_title('Guided Backprop (Explanation wrt Degree)')

model.eval()
pred = model(data.x, data.edge_index)[node_idx, :].reshape(-1, 1)

ymin, ymax = ax1.get_ylim()
xmin, xmax = ax1.get_xlim()
ax1.text(xmin, ymax - 0.1*(ymax-ymin), 'Label = {:d}'.format(data.y[node_idx].item()))
ax1.text(xmin, ymax - 0.15*(ymax-ymin), 'Pred  = {:d}'.format(pred.argmax(dim=0).item()))

plt.show()

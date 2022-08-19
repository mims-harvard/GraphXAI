import random
from networkx.classes.function import to_undirected
import networkx as nx
import torch
import matplotlib.pyplot as plt

from torch_geometric.utils import to_networkx

from graphxai.explainers import CAM, GradCAM
#from graphxai.explainers.utils.visualizations import visualize_subgraph_explanation
from graphxai.visualization.visualizations import visualize_subgraph_explanation
from graphxai.visualization.explanation_vis import visualize_node_explanation
from graphxai.gnn_models.node_classification import GCN, train, test
#from graphxai.datasets import BAShapes
from graphxai.datasets.shape_graph import ShapeGraph

from graphxai.utils import to_networkx_conv

n = 300
m = 2
num_houses = 20

# bah = BA_Houses(n, m)
# data, inhouse = bah.get_data(num_houses, multiple_features=True)

# hyp = {
#     'num_hops': 2,
#     'n': n,
#     'm': m,
#     'num_shapes': num_houses,
#     'shape_insert_strategy': 'bound_12',
#     'labeling_method': 'edge',
#     'shape_upper_bound': 1,
#     'feature_method': 'gaussian_lv'
# }

#bah = BAShapes(**hyp)
bah = ShapeGraph(model_layers = 3)
data = bah.get_graph(use_fixed_split=True)
inhouse = (data.y == 1).nonzero(as_tuple=True)[0]

model = GCN(64, input_feat = 10, classes = 2)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(1, 201):
    loss = train(model, optimizer, criterion, data)
    acc = test(model, data)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Acc: {acc:.4f}')

#random.seed(345)
node_idx = random.choice(inhouse)

model.eval()
pred = model(data.x, data.edge_index)[node_idx, :].reshape(-1, 1)
pred_class = pred.argmax(dim=0).item()
print('pred shape', pred.shape)

print('GROUND TRUTH LABEL: \t {}'.format(data.y[node_idx].item()))
print('PREDICTED LABEL   : \t {}'.format(pred.argmax(dim=0).item()))

act = lambda x: torch.argmax(x, dim=1)
cam = CAM(model, activation = act)
exp = cam.get_explanation_node(data.x, node_idx = int(node_idx), label = pred_class, edge_index = data.edge_index)

gt_exp = bah.explanations[node_idx]
# gt_exp.enc_subgraph.draw(show=True)

#t_exp.visualize_node(num_hops = 2, show=True)

print('node_imp', gt_exp.node_imp)
print('nodes for enc_subgraph', gt_exp.enc_subgraph.nodes)
print('num_hops', bah.num_hops)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

# Ground truth plot:
gt_exp.visualize_node(num_hops = bah.num_hops, graph_data = data, additional_hops = 1, heat_by_exp = True, ax = ax1)
ax1.set_title('Ground Truth')

# CAM plot:
exp.visualize_node(num_hops = bah.num_hops, graph_data = data, additional_hops = 1, heat_by_exp = True, ax = ax2)
ax2.set_title('CAM')

gcam = GradCAM(model, criterion = criterion)
exp = gcam.get_explanation_node(
                    data.x, 
                    y = data.y, 
                    node_idx = int(node_idx), 
                    edge_index = data.edge_index, 
                    average_variant=True)

#Grad-CAM plot:
exp.visualize_node(num_hops = bah.num_hops, graph_data = data, additional_hops = 1, heat_by_exp = True, ax = ax3)
ax3.set_title('Grad-CAM')

ymin, ymax = ax1.get_ylim()
xmin, xmax = ax1.get_xlim()
ax1.text(xmin, ymax - 0.1*(ymax-ymin), 'Label = {:d}'.format(data.y[node_idx].item()))
ax1.text(xmin, ymax - 0.15*(ymax-ymin), 'Pred  = {:d}'.format(pred.argmax(dim=0).item()))

plt.show()

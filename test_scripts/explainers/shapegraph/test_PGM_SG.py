import random
import torch
import matplotlib.pyplot as plt

from graphxai.explainers import PGMExplainer
from graphxai.gnn_models.node_classification.testing import GCN_3layer_basic, GIN_3layer_basic, test, train 
from graphxai.datasets.shape_graph import ShapeGraph
from graphxai.utils import aggregate_explanations

# Load dataset:
# Smaller graph is shown to work well with model accuracy, graph properties
bah = ShapeGraph(model_layers = 3, 
    num_subgraphs = 100, 
    prob_connection = 0.08, 
    subgraph_size = 13)
data = bah.get_graph(use_fixed_split=True)
inhouse = (data.y == 1).nonzero(as_tuple=True)[0]

# Test on 3-layer basic GCN, 16 hidden dim:
model = GIN_3layer_basic(16, input_feat = 11, classes = 2)

# Train the model:
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(1, 101):
    loss = train(model, optimizer, criterion, data)
    f1, acc, precision, recall, auroc, auprc = test(model, data, get_auc = True)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test F1: {f1:.4f}, Test AUROC: {auroc:.4f}')

# Get prediction of a node in the 2-house class:
model.eval()
node_idx = random.choice(inhouse).item()
pred = model(data.x, data.edge_index)[node_idx, :].reshape(-1, 1)
pred_class = pred.argmax(dim=0).item()

print('GROUND TRUTH LABEL: \t {}'.format(data.y[node_idx].item()))
print('PREDICTED LABEL   : \t {}'.format(pred.argmax(dim=0).item()))

# Set up plotting:
fig, (ax1, ax2) = plt.subplots(1, 2)

# Ground truth plot:
gt_exp = aggregate_explanations(bah.explanations[node_idx], node_level = True)
gt_exp.visualize_node(num_hops = bah.model_layers, graph_data = data, additional_hops = 0, heat_by_exp = True, ax = ax1)
ax1.set_title('Ground Truth')

# Run Explainer ----------------------------------------------------------
pgm_exp = PGMExplainer(model, explain_graph=False, p_threshold=0.1)
exp = pgm_exp.get_explanation_node(
                    node_idx,
                    data.x,  
                    edge_index = data.edge_index, 
                    top_k_nodes = 10)
# ------------------------------------------------------------------------

# Grad-CAM plot:
exp.visualize_node(num_hops = bah.model_layers, graph_data = data, additional_hops = 0, heat_by_exp = True, ax = ax2)
ax2.set_title('PGM Explainer')

# More plotting details:
ymin, ymax = ax1.get_ylim()
xmin, xmax = ax1.get_xlim()
ax1.text(xmin, ymax - 0.1*(ymax-ymin), 'Label = {:d}'.format(data.y[node_idx].item()))
ax1.text(xmin, ymax - 0.15*(ymax-ymin), 'Pred  = {:d}'.format(pred.argmax(dim=0).item()))

plt.show()

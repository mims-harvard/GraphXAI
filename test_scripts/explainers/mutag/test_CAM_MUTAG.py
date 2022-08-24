import sys
import torch
from torch_geometric.datasets import TUDataset

from graphxai.explainers import CAM, GradCAM
#from graphxai.explainers.utils.visualizations import visualize_mol_explanation
from graphxai.gnn_models.graph_classification import train, test
from graphxai.gnn_models.graph_classification.gcn import GCN_2layer, GCN_3layer
from graphxai.gnn_models.graph_classification.gin import GIN_2layer, GIN_3layer
from graphxai.datasets import MUTAG

import matplotlib.pyplot as plt

if len(sys.argv) > 1:
    seed = int(sys.argv[1])
else:
    seed = 1200

# Load data: ------------------------------------------
dataset = MUTAG(root = './data/', split_sizes = (0.8, 0.2, 0), seed = seed)
train_loader, _ = dataset.get_train_loader(batch_size = 16)
test_loader, _ = dataset.get_test_loader()

# Train GNN model -------------------------------------
model = GIN_3layer(7, 32, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(1, 101):
    train(model, optimizer, criterion, train_loader)
    f1, prec, rec, auprc, auroc = test(model, test_loader)
    print(f'Epoch: {epoch:03d}, Test F1: {f1:.4f}, Test AUROC: {auroc:.4f}')

# Choose random example in the testing data: -----------
test_data, gt_exp = dataset.get_test_w_label(label = 1) # Get positive example

# Predict on the chosen testing sample: ----------------
# Filler to get around batch variable in GNN model
null_batch = torch.zeros(1).long()
forward_kwargs = {'batch': null_batch} # Input to explainers forward methods

model.eval()    
with torch.no_grad():
    prediction = model(test_data.x, test_data.edge_index, batch = null_batch)

predicted = prediction.argmax(dim=1).item()

# Plot ground-truth explanation ------------------------
fig, (ax1, ax2) = plt.subplots(1, 2)

gt_exp.visualize_graph(ax = ax1)
ax1.set_title('Ground Truth')

# Call Explainer: --------------------------------------
act = lambda x: torch.argmax(x, dim=1)
gcam = CAM(model, activation = act)
exp = gcam.get_explanation_graph(
    x = test_data.x,
    edge_index = test_data.edge_index,
    label = test_data.y,
    forward_kwargs = forward_kwargs,
)
# ------------------------------------------------------

# Draw rest of explanations:
exp.visualize_graph(ax = ax2)
ax2.set_title('CAM')

# Draw label on the plot
ymin, ymax = ax1.get_ylim()
xmin, xmax = ax1.get_xlim()
ax1.text(xmin, ymax - 0.1*(ymax-ymin), 'Label = {:d}'.format(test_data.y.item()))
ax1.text(xmin, ymax - 0.15*(ymax-ymin), 'Pred  = {:d}'.format(predicted))

plt.show()
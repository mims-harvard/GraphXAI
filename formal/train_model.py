import ipdb
import random
import torch
from graphxai.gnn_models.node_classification.testing import GCN_3layer_basic, GIN_3layer_basic, test, train, val
from graphxai.datasets.shape_graph import ShapeGraph
from graphxai.datasets  import load_ShapeGraph

# Load ShapeGraph dataset
# Smaller graph is shown to work well with model accuracy, graph properties
bah = torch.load(open('/home/cha567/GraphXAI/data/ShapeGraph/SG_heterophilic.pickle', 'rb'))

data = bah.get_graph(use_fixed_split=True)
inhouse = (data.y == 1).nonzero(as_tuple=True)[0]

# Test on 3-layer basic GCN, 16 hidden dim:
model = GIN_3layer_basic(16, input_feat = 11, classes = 2)

# Train the model:
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

best_auroc=0
for epoch in range(1, 101):
    loss = train(model, optimizer, criterion, data)
    f1, acc, precision, recall, auroc, auprc = val(model, data, get_auc=True)
    if auroc > best_auroc:
        best_auroc = auroc
        torch.save(model.state_dict(), 'model_heterophily.pth')

    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val F1: {f1:.4f}, Val AUROC: {auroc:.4f}')

# Testing performance
f1, acc, precision, recall, auroc, auprc = test(model, data, get_auc=True)
print(f'Test F1: {f1:.4f}, Test AUROC: {auroc:.4f}')

import torch

from graphxai.datasets import load_ShapeGraph
from graphxai.gnn_models.node_classification.testing import train, test
from graphxai.gnn_models.node_classification.testing import GCN_3layer_basic, GIN_3layer_basic, GCN_4layer_basic, GAT_3layer_basic
from graphxai.gnn_models.node_classification.testing import GCN_2layer, GIN_2layer
from graphxai.gnn_models.node_classification.testing import GSAGE_3layer, JKNet_3layer, JKNet_3layer_lstm

bah = load_ShapeGraph()
data = bah.get_graph(use_fixed_split=False)

model = GIN_3layer_basic(16, input_feat = 10, classes = 2)
print(model)
print('Samples in Class 0', torch.sum(data.y == 0).item())
print('Samples in Class 1', torch.sum(data.y == 1).item())

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(1, 1001):
    loss = train(model, optimizer, criterion, data)
    #acc = test(model, data)
    f1, acc, prec, rec = test(model, data)
    #print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Acc: {acc:.4f}')
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Acc: {acc:.4f}, Test F1: {f1:.4f}')


import torch

from graphxai.datasets import load_ShapeGraph
from graphxai.datasets.shape_graph import ShapeGraph
from graphxai.gnn_models.node_classification.testing import train, test
from graphxai.gnn_models.node_classification.testing import GCN_3layer_basic, GIN_3layer_basic, GCN_4layer_basic, GAT_3layer_basic
from graphxai.gnn_models.node_classification.testing import GCN_2layer, GIN_2layer
from graphxai.gnn_models.node_classification.testing import GSAGE_3layer, JKNet_3layer, JKNet_3layer_lstm

#bah = load_ShapeGraph()
#data = bah.get_graph(use_fixed_split=False)

SG = ShapeGraph(
    model_layers = 3,
    num_subgraphs = 100,
    prob_connection = 0.08,
    subgraph_size = 10,
    class_sep = 30,
    n_informative = 2,
    verify = False,
    make_explanations = False,
    homphily_coef = -1.0
)

data = SG.get_graph()

model = GIN_3layer_basic(16, input_feat = 11, classes = 2)
print(model)
print('Samples in Class 0', torch.sum(data.y == 0).item())
print('Samples in Class 1', torch.sum(data.y == 1).item())

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(1, 1001):
    loss = train(model, optimizer, criterion, data)
    #acc = test(model, data)
    f1, acc, prec, rec, auprc, auroc = test(model, data, get_auc = True)
    #print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Acc: {acc:.4f}')
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Acc: {acc:.4f}, Test F1: {f1:.4f}, Test AUROC: {auroc:.4f}')


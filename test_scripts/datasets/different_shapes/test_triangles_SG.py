import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from graphxai.datasets import ShapeGraph
from graphxai.gnn_models.node_classification.testing import GIN_3layer_basic, GIN_2layer, train, test

# Get a triangle:
triangle = nx.Graph()
triangle.add_nodes_from([0, 1, 2])
triangle.add_edges_from([(0, 1), (1, 2), (2, 0)])

SG = ShapeGraph(
    model_layers = 3,
    shape = triangle, # NEW SHAPE
    num_subgraphs = 1300,
    prob_connection = 0.006,
    subgraph_size = 12,
    class_sep = 0.5,
    n_informative = 4,
    verify = True,
    make_explanations = True,
    homphily_coef = 1.0,
    seed = 1456,
    add_sensitive_feature = True,
    attribute_sensitive_feature = False
)

data = SG.get_graph()

SG.dump('SG_triangles.pickle')

print('Number of nodes: {}'.format(data.x.shape[0]))
print('Class imbalance:')
print('\t Y=0: {}'.format(torch.sum(data.y == 0).item()))
print('\t Y=1: {}'.format(torch.sum(data.y == 1).item()))

print('Made ShapeGraph')

degrees = sorted([d for n, d in SG.G.degree()])

variant_code = 'PA'

plt.hist(degrees, color = 'green')
plt.title('Degree Distribution - {}'.format(variant_code))
plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.show()

X = data.x.detach().clone().numpy()
Y = data.y.numpy()

print(f'Num (0): {np.sum(Y == 0)}')
print(f'Num (1): {np.sum(Y == 1)}')

# LR --------------------------------------------
parameters = {
    'C': list(np.arange(0.25, 1.5, step=0.25)),
}

lr = LogisticRegression()
clf = GridSearchCV(lr, parameters, scoring='roc_auc', verbose = 1)
clf.fit(X, Y)

print('LR Best AUROC', clf.best_score_)
print('LR Best params', clf.best_params_)

model = GIN_3layer_basic(16, input_feat = 11, classes = 2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(1, 1001):
    loss = train(model, optimizer, criterion, data)
    #acc = test(model, data)
    f1, acc, prec, rec, auprc, auroc = test(model, data, get_auc = True)
    #print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Acc: {acc:.4f}')
    if epoch % 10 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Acc: {acc:.4f}, Test F1: {f1:.4f}, Test AUROC: {auroc:.4f}')

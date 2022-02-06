import os
import torch
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from graphxai.gnn_models.node_classification.testing import train, test
from graphxai.gnn_models.node_classification.testing import GCN_3layer_basic, GIN_3layer_basic, GCN_4layer_basic, GAT_3layer_basic
from graphxai.datasets import load_ShapeGraph, ShapeGraph

root_data = os.path.join('/Users/owenqueen/Desktop/data', 'ShapeGraph')
#SG = load_ShapeGraph(number=1, root = root_data)

# SG = ShapeGraph(
#     model_layers = 3,
#     make_explanations=True,
#     num_subgraphs = 1200,
#     prob_connection = 0.0075,
#     subgraph_size = 11,
#     class_sep = 0.3,
#     n_informative = 4,
#     homophily_coef = -1,
#     n_clusters_per_class = 2,
#     seed = 1456,
#     verify = True,
#     add_sensitive_feature = True,
#     # attribute_sensitive_feature = True,
#     # sens_attribution_noise = 0.75,
# )

SG = ShapeGraph(
    model_layers = 3,
    make_explanations=True,
    num_subgraphs = 200,
    prob_connection = 0.045,
    subgraph_size = 11,
    class_sep = 0.3,
    n_informative = 4,
    homophily_coef = -1,
    n_clusters_per_class = 1,
    seed = 1456,
    verify = True,
    add_sensitive_feature = True,
    # attribute_sensitive_feature = True,
    # sens_attribution_noise = 0.75,
)

SG.dump(fname = 'SG_small_hf=-1.pickle')

# SG = ShapeGraph(
#     model_layers = 3,
#     make_explanations=False,
#     num_subgraphs = 200,
#     prob_connection = 0.045,
#     subgraph_size = 11,
#     class_sep = 0.3,
#     n_informative = 4,
#     homophily_coef = 1,
#     n_clusters_per_class = 1,
#     seed = 1456,
#     verify = False
# )

data = SG.get_graph()

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

# -----------------------------------------------
model = GIN_3layer_basic(16, input_feat = 11, classes = 2)
#model.load_state_dict()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay = 5e-5)
criterion = torch.nn.CrossEntropyLoss()

max_auroc = 0
max_f1 = 0
max_acc = 0

for epoch in trange(1, 501):
    loss = train(model, optimizer, criterion, data)
    #acc = test(model, data)
    f1, acc, prec, rec, auprc, auroc = test(model, data, get_auc = True)

    if auroc > max_auroc:
        max_auroc = auroc
        max_f1 = f1
        max_acc = acc

    #print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Acc: {acc:.4f}')
    #print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Acc: {acc:.4f}, Test F1: {f1:.4f}, Test AUROC: {auroc:.4f}')

print(f'Loss: {loss:.4f}, Test Acc: {max_acc:.4f}, Test F1: {max_f1:.4f}, Test AUROC: {max_auroc:.4f}')

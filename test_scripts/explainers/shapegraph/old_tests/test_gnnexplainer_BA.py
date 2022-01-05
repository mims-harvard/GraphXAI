import random

import ipdb
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from graphxai.gnn_models.node_classification import BA_Houses, GCN, train, test
from graphxai.datasets.feature import make_network_stats_feature
from graphxai.explainers import GNNExplainer
from graphxai.visualization import visualize_edge_explanation


# Set random seeds
seed = 0
torch.manual_seed(seed)
random.seed(seed)

n = 300
m = 2
num_houses = 20

bah = BA_Houses(n, m, seed=seed)
data, inhouse = bah.get_data(num_houses)

# Use network statistics feature
data.x, feature_imp_true, feature_names = \
    make_network_stats_feature(data.edge_index, include=['degree'],
                               num_useless_features=10)
model = GCN(16, input_feat=data.x.shape[1], classes=2)
print(model)

node_idx = int(random.choice(inhouse))


def experiment(data, plot_train=False):

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    losses = []
    test_accs = []
    for epoch in range(1, 201):
        loss = train(model, optimizer, criterion, data, losses)
        acc = test(model, data, test_accs)
    if plot_train:
        plt.plot(losses, label='training loss')
        plt.plot(test_accs, label='test acc')
        plt.legend()
        plt.show()

    def get_exp(explainer, node_idx, data):
        exp, khop_info = explainer.get_explanation_node(
            node_idx, data.x, data.edge_index, label=data.y,
            num_hops=2, explain_feature=True)
        return exp['feature_imp'], exp['edge_imp'], khop_info[0], khop_info[1]

    gnnexpr = GNNExplainer(model)
    feature_imp, edge_imp, subset, sub_edge_index = get_exp(gnnexpr, node_idx, data)

    print(f'Feature mask learned: {feature_imp}')
    # Compute feature score by F1
    feature_imp_pred = feature_imp.numpy() > 1 / data.x.shape[1]
    feature_score = f1_score(feature_imp_true, feature_imp_pred)
    print(f'F1 Feature score of gnn explainer is {feature_score}')

    sub_node_idx = -1
    for sub_idx, idx in enumerate(subset.tolist()):
        if idx == node_idx:
            sub_node_idx = sub_idx
    visualize_edge_explanation(sub_edge_index, num_nodes=len(subset),
                               node_idx=sub_node_idx, edge_imp=edge_imp)

    # Compare with ground truth unique explanation
    # Locate which house
    ipdb.set_trace()
    true_nodes, true_edges = [(nodes, edges) for nodes, edges in bah.houses if node_idx in nodes][0]
    TPs = []
    FPs = []
    FNs = []
    for i, edge in enumerate(sub_edge_index.T):
        # Restore original node numbering
        edge_ori = tuple(subset[edge].tolist())
        positive = edge_imp[i].item() > 0.8
        if positive:
            if edge_ori in true_edges:
                TPs.append(edge_ori)
            else:
                FPs.append(edge_ori)
        else:
            if edge_ori in true_edges:
                FNs.append(edge_ori)
    TP = len(TPs)
    FP = len(FPs)
    FN = len(FNs)
    edge_score = TP / (TP + FP + FN)
    print(f'TP / (TP+FP+FN) edge score of gnn explainer is {edge_score:.4f}')

experiment(data)

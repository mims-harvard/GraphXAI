import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from graphxai.gnn_models.node_classification import BA_Houses, GCN, train, test
from graphxai.datasets.feature import make_network_stats_feature
from graphxai.explainers import IntegratedGradExplainer
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
        return exp['feature_imp'], khop_info[0], khop_info[1]

    explainer = IntegratedGradExplainer(model, criterion)
    feature_imp, subset, sub_edge_index = get_exp(explainer, node_idx, data)

    print(f'Feature mask learned: {feature_imp}')
    # Compute feature score by F1
    feature_imp_pred = np.abs(feature_imp.numpy()) > 1 / data.x.shape[1]
    feature_score = f1_score(feature_imp_true, feature_imp_pred)
    print(f'F1 Feature score of IG explainer is {feature_score:.4f}')

experiment(data)

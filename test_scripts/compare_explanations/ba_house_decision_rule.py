import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from graphxai.gnn_models.node_classification import BA_Houses, GCN, train, test
from graphxai.explainers import GNNExplainer
from graphxai.visualization import visualize_edge_explanation


# Set random seeds
seed = 1
torch.manual_seed(seed)
random.seed(seed)

n = 300
m = 2
num_houses = 20

bah = BA_Houses(n, m)
data, inhouse = bah.get_data(num_houses, null_feature=True)

# Add random boolean feature and noise feature
rand_bool = torch.randint(2, (n, 1))
noise = torch.randint(2, (n, 1))
# Include null feature to ensure meaningful training
data.x = torch.hstack([data.x, rand_bool, noise])
# Apply decision rule (AND rule for unique explanation)
data.y = data.y * rand_bool.view(-1)

# Visualize the labeling of our decision rule
model = GCN(8, input_feat=4, classes=2)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

node_idx = int(random.choice(inhouse))

losses = []
test_accs = []
for epoch in range(1, 201):
    loss = train(model, optimizer, criterion, data, losses)
    acc = test(model, data, test_accs)
plt.plot(losses, label='training loss')
plt.plot(test_accs, label='test acc')
plt.legend()
plt.show()


def get_exp(explainer, node_idx, data):
    exp, khop_info = explainer.get_explanation_node(
        node_idx, data.x, data.edge_index, label=data.y,
        num_hops=2, explain_feature=True)
    return exp['feature_imp'], exp['edge_imp'], khop_info[0], khop_info[1]


gnnexpr = GNNExplainer(model, seed=seed)
feature_imp, edge_imp, subset, sub_edge_index = get_exp(gnnexpr, node_idx, data)

print(f'Feature mask learned: {feature_imp}')
# Compute feature score by F1
imp_feature_pred = feature_imp.numpy() > 1 / data.x.shape[1]
imp_feature_true = np.ones(data.x.shape[1], dtype=bool)
imp_feature_true[-noise.shape[1]:] = False
feature_score = f1_score(imp_feature_true, imp_feature_pred)
print(f'Feature score of gnn explainer is {feature_score}')

sub_node_idx = -1
for sub_idx, idx in enumerate(subset.tolist()):
    if idx == node_idx:
        sub_node_idx = sub_idx
visualize_edge_explanation(sub_edge_index, num_nodes=len(subset),
                           node_idx=sub_node_idx, edge_imp=edge_imp)

# Compare with ground truth unique explanation
# Locate which house
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
print(f'Edge score of gnn explainer is {edge_score}')

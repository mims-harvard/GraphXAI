import ipdb
import tqdm
from dig.xgraph.models import *
import torch
from torch_geometric.data import DataLoader
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.utils import convert
import os.path as osp
import os

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

# For GNNExEval paper
import torch.optim as optim
from dataset_utils import *
from metrics_edge import EdgeMetrics
from dig.xgraph.method import PGExplainer
from load_model_opt import ModelDataLoader
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from model import *
from sklearn.metrics import f1_score, roc_auc_score
from networkx.algorithms.swap import double_edge_swap as swap
from networkx.linalg.graphmatrix import adjacency_matrix as adj_mat
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# set seeds
seed = 912
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch.nn.utils import spectral_norm


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(SAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, normalize=True)
        self.conv1.aggr = 'mean'
        self.transition = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout)
        )
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, normalize=True)
        self.conv2.aggr = 'mean'
        self.transition = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout)
        )
        self.conv3 = SAGEConv(hidden_channels, hidden_channels, normalize=True)
        self.conv3.aggr = 'mean'
        self.fc = torch.nn.Linear(hidden_channels, out_channels)

        for m in self.modules():
            self.weights_init(m)

    def arguments_read(self, x, edge_index):
        return x, edge_index, _

    def weights_init(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.transition(x)
        x = self.conv2(x, edge_index)
        x = self.transition(x)
        x = self.conv3(x, edge_index)
        return F.softmax(self.fc(x), dim=-1)

    def get_emb(self, x, edge_index):
        post_conv = self.conv1(x, edge_index)
        post_conv = self.transition(post_conv)
        post_conv = self.conv2(post_conv, edge_index)
        return post_conv


def rewire_edges(x, edge_index, degree):
    # Convert to networkx graph for rewiring edges
    data = Data(x=x, edge_index=edge_index)
    G = convert.to_networkx(data, to_undirected=True)
    rewired_G = swap(G, nswap=degree, max_tries=degree * 25, seed=912)
    rewired_adj_mat = adj_mat(rewired_G)
    rewired_edge_indexes = convert.from_scipy_sparse_matrix(rewired_adj_mat)[0]
    return rewired_edge_indexes


def calculate_edge_exp_metrics(node_idx, subset, sub_edge_index, mapping, sub_x, metrics, model, device,
                               set_categorical_mask, sens_idx, degree):
    # Single node
    var = 1
    num_samples = 50
    sub_x.requires_grad = True
    model = model.to(device)
    ind = torch.where(subset == node_idx)[0].item()
    perturbed_nodes = [sub_x.clone()]
    explainer = PGExplainer(model, in_channels=512, device=device, explain_graph=False)
    _, node_explanation, _ = \
        explainer(sub_x, sub_edge_index, node_idx=ind, y=data.y[subset], top_k=5)
    node_explanation = node_explanation[0]

    top_k_features = 0.25
    top_k = np.ceil(top_k_features * sub_edge_index.shape[1])

    # Stability
    with torch.no_grad():
        cont_noise = torch.ones(sub_x.shape[1]).normal_(0, var)
        pert_sub_x = sub_x.clone()
        pert_sub_x[ind, :] += cont_noise
    pert_sub_x.requires_grad = True
    try:
        rewire_edge_index = rewire_edges(x=sub_x, edge_index=sub_edge_index, degree=degree)[:, :sub_edge_index.shape[1]]
    except:
        return None

    model = model.to(device)
    explainer = PGExplainer(model, in_channels=512, device=device, explain_graph=False)
    _, pert_node_explanation, _ = \
        explainer(pert_sub_x, sub_edge_index, node_idx=ind, y=data.y[subset], top_k=5)
    _ = torch.randn(sub_x[0, :].shape)
    pert_node_explanation = pert_node_explanation[0]
    explainer.__clear_masks__()
    _ = torch.randn(sub_edge_index[0, :].shape)
    # pert_node_explanation = explainer.explain(pert_sub_x, rewire_edge_index)

    # get predictions for original and perturbed node
    out_x = model(sub_x.to(device), sub_edge_index.to(device))
    preds_x = torch.argmax(out_x, dim=-1)

    out_pert_x = model(pert_sub_x.to(device), rewire_edge_index.to(device))
    preds_pert_x = torch.argmax(out_pert_x, dim=-1)

    if preds_x[mapping].item() == preds_pert_x[mapping].item():
        stab_score = metrics.dist_explanation(node_explanation, pert_node_explanation, top_k, ind).item()
    else:
        return None

    # Local-Group Faithfulness
    _ = torch.randn(sub_x[0, :].shape)
    explainer.__clear_masks__()
    _ = torch.randn(sub_edge_index[0, :].shape)
    # node_explanation = explainer.explain(sub_x, sub_edge_index)

    for _ in range(num_samples):
        cont_noise = torch.ones(sub_x.shape[1]).normal_(0, var)
        sub_x[ind, :] += cont_noise
        perturbed_nodes.append(sub_x.clone())
        sub_x[ind, :] -= cont_noise

    # Local-Group Faithfulness
    lg_pred_score, x_norm, lg_softmax_score, lg_norm_softmax_score = metrics.local_group_faithfulness(perturbed_nodes,
                                                                                                        node_explanation,
                                                                                                        num_samples,
                                                                                                        top_k, ind,
                                                                                                        degree)

    return lg_pred_score, stab_score, x_norm, lg_softmax_score, lg_norm_softmax_score, \
           torch.cat([node_explanation.unsqueeze(0), pert_node_explanation.unsqueeze(0)], dim=0)


dataset = 'arxiv'
# %%
model_data_loader = ModelDataLoader(dataset=dataset)

# Load data
edge_index, features, labels, idx_train, idx_val, idx_test, _, sens_idx, set_categorical_mask = model_data_loader.load_dataset()
edge_index = torch.cat((edge_index.storage.row().unsqueeze(0), edge_index.storage.col().unsqueeze(0)), dim=0)

data = Data(x=features, y=labels, edge_index=edge_index)
data = data.to(device)
data.train_mask = idx_train
data.val_mask = idx_val
data.test_mask = idx_test

# Model and optimizer and TRAIN!!!!
model = SAGE(features.shape[1], 256, 40, num_layers=3, dropout=0.5).to(device)

# model = GCN_2l(model_level='node', dim_node=features.shape[1], dim_hidden=16, num_classes=2)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
model.load_state_dict(torch.load(f'../weights_sage_{dataset}.pt'))
model.eval()

num_classes = 40  # dataset.num_classes

dataloader = DataLoader(data, batch_size=1, shuffle=False)

# set degree for rewiring
degree = 13

lg_norm_sm_faithfulness = []
stability = []
counter_fair = []
lg_fairness = []
count = 0
for idx in tqdm.tqdm(np.random.randint(0, idx_test.shape[0], idx_test.shape[0])):
    node_idx = idx_test[idx].item()
    num_hops = 1
    subset, sub_edge_index, mapping, hard_edge_mask = k_hop_subgraph(
        node_idx, num_hops, edge_index, relabel_nodes=True,
        num_nodes=features.size(0))
    sub_x = features[subset]

    sub_num_edges = sub_edge_index.size(1)
    sub_num_nodes, num_features = sub_x.size()

    # Metrics
    exp_edge_metrics = EdgeMetrics(model, labels.to(device), sub_edge_index, mapping, device)
    score = calculate_edge_exp_metrics(node_idx, subset, sub_edge_index, mapping, sub_x, exp_edge_metrics, model,
                                       device, set_categorical_mask, sens_idx, degree)

    # Add scores
    if score is not None:
        if score[1] is not None and score[2] is not None and score[3] is not None and score[4] == score[4]:
            lg_norm_sm_faithfulness.append(score[4])
            stability.append(score[1])
            count += 1

print(
        f'Faithfulness: {np.array(lg_norm_sm_faithfulness).mean():.4f}+-{np.array(lg_norm_sm_faithfulness).std() / np.sqrt(count):.4f}')
print(f'Stability: {np.array(stability).mean():.4f}+-{np.array(stability).std() / np.sqrt(count):.4f}')


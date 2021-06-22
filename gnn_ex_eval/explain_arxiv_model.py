# %%
import dgl
import ipdb
import time
import tqdm
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

import warnings

warnings.filterwarnings('ignore')

from utils import *
from models import *
from explainers import *
import networkx as nx
from metrics_edge import EdgeMetrics
from metrics_feat import FeatMetrics
from torch_geometric.data import Data
from load_model_opt import ModelDataLoader
from torch_geometric.utils import dropout_adj
from torch_geometric.utils import k_hop_subgraph
from networkx.algorithms.swap import double_edge_swap as swap
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, MessagePassing
from sklearn.metrics import f1_score, roc_auc_score
from torch_geometric.utils import dropout_adj, convert
from aif360.sklearn.metrics import consistency_score as cs
from aif360.sklearn.metrics import generalized_entropy_error as gee
from networkx.linalg.graphmatrix import adjacency_matrix as adj_mat
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator


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


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=912, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--proj_hidden', type=int, default=16,
                    help='Number of hidden units in the projection layer of encoder.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='loan',
                    choices=['arxiv', 'bail', 'credit', 'german'])
parser.add_argument("--num_heads", type=int, default=1,
                    help="number of hidden attention heads")
parser.add_argument("--num_out_heads", type=int, default=1,
                    help="number of output attention heads")
parser.add_argument("--num_layers", type=int, default=2,
                    help="number of hidden layers")
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--model', type=str, default='gcn',
                    choices=['gcn', 'gat', 'sage', 'gin', 'jk', 'infomax', 'ssf', 'rogcn', 'jaccard'])
parser.add_argument('--algo', type=str, default='grad',
                    choices=['randomf', 'randome', 'grad', 'graphlime', 'gnnex', 'ig'])
parser.add_argument('--encoder', type=str, default='gcn')
parser.add_argument('--var', type=float, default=2, help='standard deviation of the noise')
parser.add_argument('--num_samples', type=int, default=25, help='number of samples in the local-group metrics')

args = parser.parse_known_args()[0]
args.cuda = not args.no_cuda and torch.cuda.is_available()

# set seeds
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def rewire_edges(x, edge_index, degree):
    # Convert to networkx graph for rewiring edges
    data = Data(x=x, edge_index=edge_index)
    G = convert.to_networkx(data, to_undirected=True)
    rewired_G = swap(G, nswap=degree, max_tries=degree * 25, seed=912)
    rewired_adj_mat = adj_mat(rewired_G)
    rewired_edge_indexes = convert.from_scipy_sparse_matrix(rewired_adj_mat)[0]
    return rewired_edge_indexes


def calculate_feat_exp_metrics(node_idx, subset, sub_edge_index, mapping, sub_x, metrics, explainer, model, device,
                               degree, algo):

    # Single node
    top_k_features = 0.25
    if algo == 'graphlime':
        top_k_features = 0.15
    top_k = np.ceil(top_k_features * sub_x.shape[1])

    sub_x.requires_grad = True
    perturbed_nodes = [sub_x.clone()]
    if algo == 'graphlime':
        node_explanation, _, _ = explainer.explain(sub_x)
    else:
        node_explanation = explainer.explain(sub_x)
    model = model.to(device)

    model = model.to(device)
    ind = torch.where(subset == node_idx)[0].item()

    # Stability
    with torch.no_grad():
        cont_noise = torch.ones(sub_x.shape[1]).normal_(0, args.var)
        catg_temp = torch.zeros_like(cont_noise)
        pert_sub_x = sub_x.clone()
        pert_sub_x[ind, :] += cont_noise
    pert_sub_x.requires_grad = True
    try:
        rewire_edge_index = rewire_edges(x=sub_x, edge_index=sub_edge_index, degree=degree)[:, :sub_edge_index.shape[1]]
    except:
        return None

    # get predictions for original and perturbed node
    out_x = model(sub_x.to(device), sub_edge_index.to(device))
    preds_x = torch.argmax(out_x, dim=-1)

    out_pert_x = model(pert_sub_x.to(device), rewire_edge_index.to(device))
    preds_pert_x = torch.argmax(out_pert_x, dim=-1)

    if preds_x[mapping].item() == preds_pert_x[mapping].item():
        _ = torch.randn(sub_edge_index[0, :].shape)
        _ = torch.randn(sub_x[0, :].shape)
        if algo == 'graphlime':
            pert_node_explanation, _, _ = explainer.explain(pert_sub_x)
        else:
            pert_node_explanation = explainer.explain(pert_sub_x)
        if algo in ['grad', 'graphlime', 'ig']:
            _ = torch.randn(sub_x[0, :].shape)
        _ = torch.randn(sub_edge_index[0, :].shape)
        model = model.to(device)
        stab_score = metrics.dist_explanation(node_explanation, pert_node_explanation, top_k, ind)
    else:
        return None

    # Local-Group Faithfulness
    if algo in ['grad', 'graphlime', 'ig']:
        _ = torch.randn(sub_x[0, :].shape)
    _ = torch.randn(sub_edge_index[0, :].shape)

    for jjj in range(args.num_samples):
        cont_noise = torch.ones(sub_x.shape[1]).normal_(0, args.var)
        sub_x[ind, :] += cont_noise
        perturbed_nodes.append(sub_x.clone())
        sub_x[ind, :] -= cont_noise

    # Local-Group Faithfulness
    _, _, _, lg_norm_softmax_score = metrics.local_group_faithfulness(perturbed_nodes,
                                                                      node_explanation,
                                                                      args.num_samples,
                                                                      top_k, ind,
                                                                      degree)

    return stab_score, lg_norm_softmax_score


def calculate_edge_exp_metrics(node_idx, subset, sub_edge_index, mapping, sub_x, metrics, explainer, model, device,
                               degree, algo):
    # Single node
    top_k_features = 0.25
    top_k = np.ceil(top_k_features * sub_edge_index.shape[1])
    sub_x.requires_grad = True
    model = model.to(device)
    ind = torch.where(subset == node_idx)[0].item()
    perturbed_nodes = [sub_x.clone()]
    _ = torch.randn(sub_x[0, :].shape)
    node_explanation = explainer.explain(sub_x, sub_edge_index)
    model = model.to(device)
    if algo in ['gnnex']:
        _ = torch.randn(sub_edge_index[0, :].shape)

    # Stability
    with torch.no_grad():
        pert_sub_x = sub_x.clone()
        cont_noise = torch.ones(sub_x.shape[1]).normal_(0, args.var)
        pert_sub_x[ind, :] += cont_noise
    pert_sub_x.requires_grad = True
    try:
        rewire_edge_index = rewire_edges(x=sub_x, edge_index=sub_edge_index, degree=degree)[:, :sub_edge_index.shape[1]]
    except:
        return None

    # get predictions for original and perturbed node
    out_x = model(sub_x.to(device), sub_edge_index.to(device))
    preds_x = torch.argmax(out_x, dim=-1)

    out_pert_x = model(pert_sub_x.to(device), rewire_edge_index.to(device))
    preds_pert_x = torch.argmax(out_pert_x, dim=-1)

    if preds_x[mapping].item() == preds_pert_x[mapping].item():
        explainer.__set_masks__()
        if algo in ['randome']:
            _ = torch.randn(sub_edge_index[0, :].shape)
            _ = torch.randn(sub_x[0, :].shape)
        model = model.to(device)
        _ = torch.randn(sub_x[0, :].shape)
        pert_node_explanation = explainer.explain(pert_sub_x, rewire_edge_index)
        if algo in ['gnnex']:
            _ = torch.randn(sub_edge_index[0, :].shape)
        stab_score = metrics.dist_explanation(node_explanation, pert_node_explanation, top_k, ind).item()
    else:
        return None

    # Local-Group Faithfulness
    for jjj in range(args.num_samples):
        cont_noise = torch.ones(sub_x.shape[1]).normal_(0, args.var)
        sub_x[ind, :] += cont_noise
        perturbed_nodes.append(sub_x.clone())
        sub_x[ind, :] -= cont_noise

    # Local-Group Faithfulness
    _, _, _, lg_norm_softmax_score = metrics.local_group_faithfulness(perturbed_nodes,
                                                                      node_explanation,
                                                                      args.num_samples,
                                                                      top_k, ind,
                                                                      degree)

    return stab_score, lg_norm_softmax_score


def train(model, adj, features, idx_train, labels, optimizer):
    optimizer.zero_grad()
    out = model(features, adj)[idx_train]
    loss = F.nll_loss(out, labels[idx_train])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, adj, features, idx_train, idx_val, idx_test, labels, evaluator):
    model.eval()

    out = model(features, adj)
    y_pred = out.argmax(dim=-1, keepdim=True)
    train_acc = evaluator.eval({
        'y_true': labels[idx_train].unsqueeze(-1),
        'y_pred': y_pred[idx_train],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': labels[idx_val].unsqueeze(-1),
        'y_pred': y_pred[idx_val],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': labels[idx_test].unsqueeze(-1),
        'y_pred': y_pred[idx_test],
    })['acc']

    return train_acc, valid_acc, test_acc


def train_model(args, adj, features, labels, idx_train, idx_val, idx_test, model):
    evaluator = Evaluator(name='ogbn-arxiv')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_acc = 0
    for epoch in range(1, 1 + args.epochs):
        loss = train(model, adj, features, idx_train, labels, optimizer)
        result = test(model, adj, features, idx_train, idx_val, idx_test, labels, evaluator)
        train_acc, valid_acc, test_acc = result
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), f'weights_{args.model}_{args.dataset}.pt')
        if epoch % 50 == 0:
            print(f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * train_acc:.2f}%, '
                  f'Valid: {100 * valid_acc:.2f}% '
                  f'Test: {100 * test_acc:.2f}%')


def main():
    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # %%
    model_data_loader = ModelDataLoader(args)

    # Load data
    edge_index, features, labels, idx_train, idx_val, idx_test, _, sens_idx, set_categorical_mask = model_data_loader.load_dataset()
    edge_index = torch.cat((edge_index.storage.row().unsqueeze(0), edge_index.storage.col().unsqueeze(0)), dim=0)

    # initialize model
    model = SAGE(features.shape[1], args.hidden, 40, num_layers=3, dropout=args.dropout).to(device)
    # train_model(args, edge_index.to(device), features.to(device), labels.to(device), idx_train, idx_val, idx_test, model)

    # Model and optimizer
    model.load_state_dict(torch.load(f'weights_{args.model}_{args.dataset}.pt'))
    model.eval()
    
    with torch.no_grad():
        model.eval()
        from ogb.nodeproppred import Evaluator
        evaluator = Evaluator(name='ogbn-arxiv')
        out = model(features.to(device), edge_index.to(device))
        y_pred = out.argmax(dim=-1, keepdim=True)
        test_acc = evaluator.eval({
            'y_true': labels[idx_test].unsqueeze(-1),
            'y_pred': y_pred[idx_test],
        })['acc']
        print(f'Testing Acc.: {100*test_acc:.2f}')

    # set degree for rewiring
    degree = 13

    count = 0
    stability = []
    lg_norm_sm_faithfulness = []
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
        exp_feat_metrics = FeatMetrics(model, labels.to(device), sub_edge_index, mapping, device)
        exp_edge_metrics = EdgeMetrics(model, labels.to(device), sub_edge_index, mapping, args.algo, device)

        # Explanation Algorithm
        if args.algo == 'grad':
            explainer = VanillaGrad(model, labels.to(device), sub_x, sub_edge_index, mapping, node_idx, subset, device)
            _ = torch.randn(sub_edge_index[0, :].shape)
            _ = torch.randn(sub_x[0, :].shape)
            score = calculate_feat_exp_metrics(node_idx, subset, sub_edge_index, mapping, sub_x, exp_feat_metrics,
                                               explainer, model, device, degree, args.algo)
        if args.algo == 'ig':
            explainer = IntegratedGrad(model, labels.to(device), sub_x, sub_edge_index, mapping, node_idx, subset, device)
            _ = torch.randn(sub_edge_index[0, :].shape)
            _ = torch.randn(sub_x[0, :].shape)
            score = calculate_feat_exp_metrics(node_idx, subset, sub_edge_index, mapping, sub_x, exp_feat_metrics,
                                               explainer, model, device, degree, args.algo)
        elif args.algo == 'graphlime':
            explainer = GLIME(model, sub_x, torch.where(subset == node_idx)[0].item(), sub_edge_index, hop=2, rho=0.0,
                              device=device)
            _ = torch.randn(sub_edge_index[0, :].shape)
            _ = torch.randn(sub_x[0, :].shape)
            score = calculate_feat_exp_metrics(node_idx, subset, sub_edge_index, mapping, sub_x, exp_feat_metrics,
                                               explainer, model, device, degree, args.algo)
        elif args.algo == 'randomf':
            explainer = RandomFeatures()
            _ = torch.randn(sub_edge_index[0, :].shape)
            _ = torch.randn(sub_x[0, :].shape)
            score = calculate_feat_exp_metrics(node_idx, subset, sub_edge_index, mapping, sub_x, exp_feat_metrics,
                                               explainer, model, device, degree, args.algo)

        elif args.algo == 'randome':
            explainer = RandomEdges()
            _ = torch.randn(sub_edge_index[0, :].shape)
            _ = torch.randn(sub_x[0, :].shape)
            score = calculate_edge_exp_metrics(node_idx, subset, sub_edge_index, mapping, sub_x, exp_edge_metrics,
                                               explainer, model, device, degree, args.algo)

        elif args.algo == 'gnnex':
            explainer = GNNExplainer(model, labels.to(device), sub_x, sub_edge_index, mapping, node_idx, subset,
                                     sub_num_nodes, sub_num_edges, num_features, device)
            explainer.__set_masks__()
            score = calculate_edge_exp_metrics(node_idx, subset, sub_edge_index, mapping, sub_x, exp_edge_metrics,
                                               explainer, model, device, degree, args.algo)

        # Add scores
        if score is not None:
            if score[0] is not None and score[1] is not None:
                lg_norm_sm_faithfulness.append(score[1])
                stability.append(score[0])
                count+=1

    # Generate report
    print(
        f'Faithfulness: {np.array(lg_norm_sm_faithfulness).mean():.4f}+-{np.array(lg_norm_sm_faithfulness).std() / np.sqrt(count):.4f}')
    print(f'Stability: {np.array(stability).mean():.4f}+-{np.array(stability).std()/np.sqrt(count):.4f}')


if __name__ == "__main__":
    main()

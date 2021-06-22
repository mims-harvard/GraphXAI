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

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=912, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
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
                               set_categorical_mask, sens_idx, degree, algo):

    # Single node
    sub_x.requires_grad = True
    top_k_features = 0.25
    top_k = np.ceil(top_k_features * sub_x.shape[1])

    model = model.to(device)
    ind = torch.where(subset == node_idx)[0].item()

    # Local-Group Faithfulness
    perturbed_nodes = [sub_x.clone()]
    if algo == 'graphlime':
        node_explanation, _, _ = explainer.explain(sub_x)
    else:
        node_explanation = explainer.explain(sub_x)
    model = model.to(device)
    if algo in ['grad', 'graphlime', 'ig']:
        _ = torch.randn(sub_x[0, :].shape)
    _ = torch.randn(sub_edge_index[0, :].shape)

    # Stability
    with torch.no_grad():
        cont_noise = torch.ones(sub_x.shape[1]).normal_(0, args.var).mul(1 - set_categorical_mask[:, 1])
        catg_temp = torch.zeros_like(cont_noise)
        pert_sub_x = sub_x.clone()
        for ii in range(catg_temp.shape[0]):
            # categorical flip with some prob p
            if set_categorical_mask[ii][1] == 1 and torch.rand(1) <= 0.1:
                catg_temp[ii] = sub_x[ind, ii]
                pert_sub_x[ind, ii] = torch.randint(torch.unique(sub_x[:, ii].detach())[0].int().item(),
                                                    torch.unique(sub_x[:, ii].detach())[-1].int().item() + 1, (1,))

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
        stab_score = None
        pert_node_explanation = torch.zeros_like(node_explanation)

    # Counterfactual Fairness
    with torch.no_grad():
        counter_sub_x = sub_x.clone()
        counter_sub_x[ind, sens_idx] = 1 - counter_sub_x[ind, sens_idx]
    counter_sub_x.requires_grad = True

    # get predictions for original and perturbed node
    out_x = model(sub_x.to(device), sub_edge_index.to(device))
    preds_x = torch.argmax(out_x, dim=-1)

    out_counter_x = model(counter_sub_x.to(device), sub_edge_index.to(device))
    preds_counter_x = torch.argmax(out_counter_x, dim=-1)

    if preds_x[mapping].item() == preds_counter_x[mapping].item():
        _ = torch.randn(sub_edge_index[0, :].shape)
        _ = torch.randn(sub_x[0, :].shape)

        if algo == 'graphlime':
            counter_node_explanation, _, _ = explainer.explain(counter_sub_x)
        else:
            counter_node_explanation = explainer.explain(counter_sub_x)

        if algo in ['grad', 'graphlime', 'ig']:
            _ = torch.randn(sub_x[0, :].shape)
        _ = torch.randn(sub_edge_index[0, :].shape)

        counter_fair_score = metrics.dist_explanation(node_explanation, counter_node_explanation, top_k, ind)
    else:
        counter_fair_score = None
        counter_node_explanation = torch.zeros_like(node_explanation)

    for jjj in range(args.num_samples):
        cont_noise = torch.ones(sub_x.shape[1]).normal_(0, args.var).mul(1 - set_categorical_mask[:, 1])
        catg_temp = torch.zeros_like(cont_noise)

        for ii in range(catg_temp.shape[0]):
            # categorical flip with some prob p  and torch.rand(1) <= 0.1:
            if set_categorical_mask[ii][1] == 1 and torch.rand(1) <= 0.1:
                catg_temp[ii] = sub_x[ind, ii]
                sub_x[ind, ii] = torch.randint(torch.unique(sub_x[:, ii].detach())[0].int().item(),
                                               torch.unique(sub_x[:, ii].detach())[-1].int().item() + 1, (1,))
        sub_x[ind, :] += cont_noise
        perturbed_nodes.append(sub_x.clone())
        sub_x[ind, :] -= cont_noise

        for ii in range(catg_temp.shape[0]):
            if set_categorical_mask[ii][1] == 1:
                sub_x[ind, ii] = catg_temp[ii]

    # Local-Group Faithfulness
    lg_pred_score, x_norm, lg_softmax_score, lg_norm_softmax_score = metrics.local_group_faithfulness(perturbed_nodes,
                                                                                                      node_explanation,
                                                                                                      args.num_samples,
                                                                                                      top_k, ind,
                                                                                                      degree)

    # Local-Group Fairness
    lg_fair_score, fairness_bound = metrics.local_group_fairness(perturbed_nodes, subset, node_explanation,
                                                                 args.num_samples,
                                                                 top_k, ind,
                                                                 sens_idx, degree)

    return lg_pred_score, stab_score, counter_fair_score, lg_fair_score, x_norm, lg_softmax_score, \
           lg_norm_softmax_score, torch.cat([node_explanation.unsqueeze(0), pert_node_explanation.unsqueeze(0),
                                             counter_node_explanation.unsqueeze(0)], dim=0), fairness_bound


def calculate_edge_exp_metrics(node_idx, subset, sub_edge_index, mapping, sub_x, metrics, explainer, model, device,
                               set_categorical_mask, sens_idx, degree, algo):
    # Single node
    sub_x.requires_grad = True
    top_k_features = 0.25
    top_k = np.ceil(top_k_features * sub_edge_index.shape[1])
    model = model.to(device)
    ind = torch.where(subset == node_idx)[0].item()

    # Local-Group Faithfulness
    perturbed_nodes = [sub_x.clone()]
    _ = torch.randn(sub_x[0, :].shape)
    node_explanation = explainer.explain(sub_x, sub_edge_index)
    model = model.to(device)
    if algo in ['gnnex']:
        _ = torch.randn(sub_edge_index[0, :].shape)

    # Stability
    with torch.no_grad():
        cont_noise = torch.ones(sub_x.shape[1]).normal_(0, args.var).mul(1 - set_categorical_mask[:, 1])
        catg_temp = torch.zeros_like(cont_noise)
        pert_sub_x = sub_x.clone()
        for ii in range(catg_temp.shape[0]):
            # categorical flip with some prob p
            if set_categorical_mask[ii][1] == 1 and torch.rand(1) <= 0.1:
                catg_temp[ii] = sub_x[ind, ii]
                pert_sub_x[ind, ii] = torch.randint(torch.unique(sub_x[:, ii].detach())[0].int().item(),
                                                    torch.unique(sub_x[:, ii].detach())[-1].int().item() + 1, (1,))
        pert_sub_x[ind, :] += cont_noise  # torch.ones(sub_x.shape[1]).normal_(0, args.var)
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
        stab_score = None
        pert_node_explanation = torch.zeros_like(node_explanation)

    # Counterfactual Fairness
    with torch.no_grad():
        counter_sub_x = sub_x.clone()
        counter_sub_x[ind, sens_idx] = 1 - counter_sub_x[ind, sens_idx]
    counter_sub_x.requires_grad = True

    # get predictions for original and perturbed node
    out_x = model(sub_x.to(device), sub_edge_index.to(device))
    preds_x = torch.argmax(out_x, dim=-1)

    out_counter_x = model(counter_sub_x.to(device), sub_edge_index.to(device))
    preds_counter_x = torch.argmax(out_counter_x, dim=-1)

    if preds_x[mapping].item() == preds_counter_x[mapping].item():
        explainer.__set_masks__()
        if algo in ['randome']:
            _ = torch.randn(sub_edge_index[0, :].shape)
            _ = torch.randn(sub_x[0, :].shape)
        _ = torch.randn(sub_x[0, :].shape)
        counter_node_explanation = explainer.explain(counter_sub_x.to(device), sub_edge_index.to(device))
        if algo in ['gnnex']:
            _ = torch.randn(sub_edge_index[0, :].shape)
        counter_fair_score = metrics.dist_explanation(node_explanation, counter_node_explanation, top_k, ind).item()
    else:
        counter_fair_score = None
        counter_node_explanation = torch.zeros_like(node_explanation)

    for jjj in range(args.num_samples):
        cont_noise = torch.ones(sub_x.shape[1]).normal_(0, args.var).mul(1 - set_categorical_mask[:, 1])
        catg_temp = torch.zeros_like(cont_noise)

        for ii in range(catg_temp.shape[0]):
            # categorical flip with some prob p  and torch.rand(1) <= 0.1:
            if set_categorical_mask[ii][1] == 1 and torch.rand(1) <= 0.1:
                catg_temp[ii] = sub_x[ind, ii]
                sub_x[ind, ii] = torch.randint(torch.unique(sub_x[:, ii].detach())[0].int().item(),
                                               torch.unique(sub_x[:, ii].detach())[-1].int().item() + 1, (1,))
        sub_x[ind, :] += cont_noise
        perturbed_nodes.append(sub_x.clone())
        sub_x[ind, :] -= cont_noise

        for ii in range(catg_temp.shape[0]):
            if set_categorical_mask[ii][1] == 1:
                sub_x[ind, ii] = catg_temp[ii]

    # Local-Group Faithfulness
    lg_pred_score, x_norm, lg_softmax_score, lg_norm_softmax_score = metrics.local_group_faithfulness(perturbed_nodes,
                                                                                                      node_explanation,
                                                                                                      args.num_samples,
                                                                                                      top_k, ind,
                                                                                                      degree)

    # Local-Group Fairness
    lg_fair_score, fairness_bound = metrics.local_group_fairness(perturbed_nodes, subset, node_explanation,
                                                                 args.num_samples,
                                                                 top_k, ind,
                                                                 sens_idx, degree)

    return lg_pred_score, stab_score, counter_fair_score, lg_fair_score, x_norm, lg_softmax_score, lg_norm_softmax_score, torch.cat(
        [node_explanation.unsqueeze(0), pert_node_explanation.unsqueeze(0), counter_node_explanation.unsqueeze(0)],
        dim=0), fairness_bound


def main():
    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # %%
    model_data_loader = ModelDataLoader(args)

    # Load data
    adj, features, labels, _, _, idx_test, _, sens_idx, set_categorical_mask = model_data_loader.load_dataset()
    edge_index = convert.from_scipy_sparse_matrix(adj)[0]

    # Model and optimizer
    model, optimizer = model_data_loader.load_model_optim(nfeat=features.shape[1],
                                                          num_class=labels.unique().shape[0])  # -1)
    model = model.to(device)
    model.load_state_dict(torch.load(f'weights_{args.model}_{args.dataset}.pt'))
    model.eval()
    print(model)

    output = model(features.to(device), edge_index.to(device))
    output_preds = torch.argmax(output, dim=-1)
    auc_roc_test = roc_auc_score(labels.cpu().numpy()[idx_test.cpu()],
                                 output.detach().cpu().numpy()[idx_test.cpu()][:, 1])
    f1_s = f1_score(labels[idx_test].cpu().numpy(), output_preds[idx_test].cpu().numpy())
    print("The AUCROC of estimator: {:.4f}".format(auc_roc_test))
    print(f'F1-score: {f1_s}')

    # set degree for rewiring
    if args.dataset == 'german':
        degree = 45
    elif args.dataset == 'bail':
        degree = 34
    else:
        degree = 96

    lg_pred_faithfulness = []
    lg_sm_faithfulness = []
    lg_norm_sm_faithfulness = []
    stability = []
    counter_fair = []
    lg_fairness = []
    explanations = []
    x_norm = []
    fairness_bound = []
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
        exp_feat_metrics = FeatMetrics(model, labels.to(device), sub_edge_index, mapping, device)
        exp_edge_metrics = EdgeMetrics(model, labels.to(device), sub_edge_index, mapping, args.algo, device)

        # Explanation Algorithm
        if args.algo == 'grad':
            explainer = VanillaGrad(model, labels.to(device), sub_x, sub_edge_index, mapping, node_idx, subset, device)
            _ = torch.randn(sub_edge_index[0, :].shape)
            _ = torch.randn(sub_x[0, :].shape)
            score = calculate_feat_exp_metrics(node_idx, subset, sub_edge_index, mapping, sub_x, exp_feat_metrics,
                                               explainer, model, device, set_categorical_mask, sens_idx, degree, args.algo)
        if args.algo == 'ig':
            explainer = IntegratedGrad(model, labels.to(device), sub_x, sub_edge_index, mapping, node_idx, subset, device)
            _ = torch.randn(sub_edge_index[0, :].shape)
            _ = torch.randn(sub_x[0, :].shape)
            score = calculate_feat_exp_metrics(node_idx, subset, sub_edge_index, mapping, sub_x, exp_feat_metrics,
                                               explainer, model, device, set_categorical_mask, sens_idx, degree, args.algo)
        elif args.algo == 'graphlime':
            explainer = GLIME(model, sub_x, torch.where(subset == node_idx)[0].item(), sub_edge_index, hop=1, rho=0.01,
                              device=device)
            _ = torch.randn(sub_edge_index[0, :].shape)
            _ = torch.randn(sub_x[0, :].shape)
            score = calculate_feat_exp_metrics(node_idx, subset, sub_edge_index, mapping, sub_x, exp_feat_metrics,
                                               explainer, model, device, set_categorical_mask, sens_idx, degree, args.algo)
        elif args.algo == 'randomf':
            explainer = RandomFeatures()
            _ = torch.randn(sub_edge_index[0, :].shape)
            _ = torch.randn(sub_x[0, :].shape)
            score = calculate_feat_exp_metrics(node_idx, subset, sub_edge_index, mapping, sub_x, exp_feat_metrics,
                                               explainer, model, device, set_categorical_mask, sens_idx, degree, args.algo)

        elif args.algo == 'randome':
            explainer = RandomEdges()
            _ = torch.randn(sub_edge_index[0, :].shape)
            _ = torch.randn(sub_x[0, :].shape)
            score = calculate_edge_exp_metrics(node_idx, subset, sub_edge_index, mapping, sub_x, exp_edge_metrics,
                                               explainer, model, device, set_categorical_mask, sens_idx, degree, args.algo)

        elif args.algo == 'gnnex':
            explainer = GNNExplainer(model, labels.to(device), sub_x, sub_edge_index, mapping, node_idx, subset,
                                     sub_num_nodes, sub_num_edges, num_features, device)
            explainer.__set_masks__()
            score = calculate_edge_exp_metrics(node_idx, subset, sub_edge_index, mapping, sub_x, exp_edge_metrics,
                                               explainer, model, device, set_categorical_mask, sens_idx, degree, args.algo)

        # Add scores
        if score is not None:
            if score[1] is not None and score[2] is not None and score[3] is not None and score[4] == score[4]:
                lg_pred_faithfulness.append(score[0])
                lg_sm_faithfulness.append(score[5])
                lg_norm_sm_faithfulness.append(score[6])
                x_norm.append(score[4])
                lg_fairness.append(score[3])
                stability.append(score[1])
                counter_fair.append(score[2])
                explanations.append(score[7].cpu().detach().numpy())
                fairness_bound.append(score[8])
                count += 1

    # Generate report
    print(
        f'Faithfulness: {np.array(lg_norm_sm_faithfulness).mean():.4f}+-{np.array(lg_norm_sm_faithfulness).std() / np.sqrt(count):.4f}')
    print(f'Stability: {np.array(stability).mean():.4f}+-{np.array(stability).std()/np.sqrt(count):.4f}')
    print(f'Counterfactual Fairnss: {np.array(counter_fair).mean():.4f}+-{np.array(counter_fair).std()/np.sqrt(count):.4f}')
    print(f'Fairness: {np.array(lg_fairness).mean():.4f}+-{np.array(lg_fairness).std()/np.sqrt(count):.4f}')


if __name__ == "__main__":
    main()

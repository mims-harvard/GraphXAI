import random, sys
from typing import Union, List
import os
from networkx.classes.function import to_undirected
import networkx as nx
import ipdb
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from numpy import ndarray
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data

from graphxai.explainers import CAM, GradCAM, GNNExplainer
# from graphxai.explainers.utils.visualizations import visualize_subgraph_explanation
from graphxai.visualization.visualizations import visualize_subgraph_explanation
from graphxai.visualization.explanation_vis import visualize_node_explanation
from graphxai.gnn_models.node_classification import GCN, train, test
from graphxai.gnn_models.node_classification.testing import GCN_3layer_basic, GIN_3layer_basic

from graphxai.gnn_models.node_classification import GCN, train, test
from graphxai.gnn_models.node_classification.testing import GCN_3layer_basic, train, test

from graphxai.datasets.shape_graph import ShapeGraph
from graphxai.utils import to_networkx_conv, Explanation, distance
from graphxai.utils.perturb import rewire_edges, perturb_node_features


device = "cuda" if torch.cuda.is_available() else "cpu"

def graph_exp_acc(gt_exp: List[Explanation], generated_exp: Explanation, node_thresh_factor = 0.1) -> float:
    '''
    Args:
        gt_exp (Explanation): Ground truth explanation from the dataset.
        generated_exp (Explanation): Explanation output by an explainer.
    '''

    EPS = 1e-09
    JAC_feat = None
    JAC_node = None
    JAC_edge = None

    # Accessing the enclosing subgraph. Will be the same for both explanation.:
    exp_subgraph = generated_exp.enc_subgraph
    if generated_exp.feature_imp is not None:
        JAC_feat = []
        thresh_feat = generated_exp.feature_imp.mean()
        for exp in gt_exp:
            TPs = []
            FPs = []
            FNs = []
            true_feat = torch.where(exp.feature_imp == 1)[0]
            for i, feat in enumerate(exp.feature_imp):
                # Restore original feature numbering
                positive = generated_exp.feature_imp[i].item() > thresh_feat
                if positive:
                    if i in true_feat:
                        TPs.append(generated_exp.feature_imp[i])
                    else:
                        FPs.append(generated_exp.feature_imp[i])
                else:
                    if i in true_feat:
                        FNs.append(generated_exp.feature_imp[i])

            TP = len(TPs)
            FP = len(FPs)
            FN = len(FNs)
            JAC_feat.append(TP / (TP + FP + FN + EPS))

        JAC_feat = max(JAC_feat)

    if generated_exp.node_imp is not None:
        JAC_node = []
        thresh_node = node_thresh_factor*generated_exp.node_imp.max()
        for exp in gt_exp:
            TPs = []
            FPs = []
            FNs = []
            relative_positives = (exp.node_imp == 1).nonzero(as_tuple=True)[0]
            true_nodes = [exp.enc_subgraph.nodes[i].item() for i in relative_positives]
            for i, node in enumerate(exp_subgraph.nodes):
                # Restore original node numbering
                positive = generated_exp.node_imp[i].item() > thresh_node
                if positive:
                    if node in true_nodes:
                        TPs.append(node)
                    else:
                        FPs.append(node)
                else:
                    if node in true_nodes:
                        FNs.append(node)
            TP = len(TPs)
            FP = len(FPs)
            FN = len(FNs)
            JAC_node.append(TP / (TP + FP + FN + EPS))

        JAC_node = max(JAC_node)

    if generated_exp.edge_imp is not None:
    #if False:
        JAC_edge = []
        for exp in gt_exp:
            TPs = []
            FPs = []
            FNs = []
            true_edges = torch.where(exp.edge_imp == 1)[0]
            for edge in range(exp.edge_imp.shape[0]):
                if generated_exp.edge_imp[edge]:
                    if edge in true_edges:
                        TPs.append(edge)
                    else:
                        FPs.append(edge)
                else:
                    if edge in true_edges:
                        FNs.append(edge)
            TP = len(TPs)
            FP = len(FPs)
            FN = len(FNs)
            JAC_edge.append(TP / (TP + FP + FN + EPS))

        JAC_edge = max(JAC_edge)

    return [JAC_feat, JAC_node, JAC_edge]


def graph_exp_faith(generated_exp: Explanation, 
        shape_graph: ShapeGraph, 
        model, 
        sens_idx: List[int]= [], 
        top_k: float = 0.25) -> float:
    '''
    Args:
        gt_exp (Explanation): Ground truth explanation from the dataset.
        generated_exp (Explanation): Explanation output by an explainer.
    '''

    GEF_feat = None
    GEF_node = None
    GEF_edge = None

    # Accessing the enclosing subgraph. Will be the same for both explanation.:
    exp_subgraph = generated_exp.enc_subgraph

    data = shape_graph.get_graph(use_fixed_split=True)
    X = data.x.to(device)
    Y = data.y.to(device)
    EIDX = data.edge_index.to(device)

    # Getting the softmax vector for the original graph
    org_vec = model(X, EIDX)[generated_exp.node_idx]
    org_softmax = F.softmax(org_vec, dim=-1)

    if generated_exp.feature_imp is not None:
        # Identifying the top_k features in the node attribute feature vector
        top_k_features = generated_exp.feature_imp.topk(int(generated_exp.feature_imp.shape[0] * top_k))[1]

        # Getting the softmax vector for the perturbed graph
        pert_x = X.clone()

        # Perturbing the unimportant node feature indices using gaussian noise
        rem_features = torch.Tensor(
            [i for i in range(X.shape[1]) if i not in top_k_features]).long()

        pert_x[generated_exp.node_idx, rem_features] = perturb_node_features(x=pert_x, node_idx=generated_exp.node_idx, pert_feat=rem_features, bin_dims=sens_idx, device = device)

        pert_vec = model(pert_x.to(device), EIDX)[generated_exp.node_idx]
        pert_softmax = F.softmax(pert_vec, dim=-1)
        GEF_feat = 1 - torch.exp(-F.kl_div(org_softmax.log(), pert_softmax, None, None, 'sum')).item()

    if generated_exp.node_imp is not None:

        # Identifying the top_k nodes in the explanation subgraph
        top_k_nodes = generated_exp.node_imp.topk(int(generated_exp.node_imp.shape[0] * top_k))[1]

        rem_nodes = []
        for node in range(generated_exp.node_imp.shape[0]):
            if node not in top_k_nodes:
                rem_nodes.append([k for k, v in generated_exp.node_reference.items() if v == node][0])

        # Getting the softmax vector for the perturbed graph
        pert_x = X.clone()

        # Removing the unimportant nodes by masking
        pert_x[rem_nodes] = torch.zeros_like(pert_x[rem_nodes]).to(device)
        pert_vec = model(pert_x, EIDX)[generated_exp.node_idx]
        pert_softmax = F.softmax(pert_vec, dim=-1)
        GEF_node = 1 - torch.exp(-F.kl_div(org_softmax.log(), pert_softmax, None, None, 'sum')).item()

    if generated_exp.edge_imp is not None:
        subgraph_edges = torch.where(generated_exp.enc_subgraph.edge_mask == True)[0].to(device)
        #lookup_subgraph_edges = set(subgraph_edges.tolist())
        # Get the list of all edges that we need to keep

        # Identifying the top_k edges in the explanation subgraph
        top_k_edges = generated_exp.edge_imp.topk(int(generated_exp.edge_imp.shape[0] * top_k))[1]
        topk_lookup = set(top_k_edges.tolist())

        subgraph_mask = torch.zeros(subgraph_edges.shape[0]).bool()
        subgraph_mask[top_k_edges] = True

        # Bigger indices:
        to_remove = subgraph_edges[subgraph_mask]
        wholeEIDX_mask = torch.ones(EIDX.shape[1]).bool()
        wholeEIDX_mask[to_remove] = False

        edge_index = EIDX[:,wholeEIDX_mask] # Use boolean mask

        # on_subgraph_mask = [True if e in topk_lookup  for e in subgraph_edges.tolist()]

        
        # wholeEIDX_mask = 

        # Only defined on the subgraph
        #lookup_top_k_edges = 

        

        # keep_edges = [] 
        # for i in range(EIDX.shape[1]):
        #     if i in lookup_subgraph_edges and top_k_edges:
        #         continue
        #     # if i in subgraph_edges and generated_exp.edge_imp[(subgraph_edges == i).nonzero(as_tuple=True)[0]]==0:
        #     #     continue
        #     else:
        #         keep_edges.append(i)

        # Get new edge_index
        #edge_index = EIDX[:, keep_edges]
                    
        # Getting the softmax vector for the perturbed graph
        pert_vec = model(X, edge_index.to(device))[generated_exp.node_idx]
        pert_softmax = F.softmax(pert_vec, dim=-1)        
        GEF_edge = 1 - torch.exp(-F.kl_div(org_softmax.log(), pert_softmax, None, None, 'sum')).item()

    return [GEF_feat, GEF_node, GEF_edge]


def calculate_delta(x, edge_index, train_set, label, sens_idx, model = None, rep='softmax', dist_norm=2, device = 'cpu'):

    x = x.to(device)
    edge_index = edge_index.to(device)

    delta_softmax, delta_L1, delta_L2, delta_Lfinal = [], [], [], []

    for n_id in train_set[torch.randperm(train_set.size()[0])][:1000]:
        try:
            pert_edge_index = rewire_edges(edge_index, node_idx=n_id.item(), num_nodes=1).to(device)
        except:
            continue
        pert_x = x.clone()
        pert_x[n_id] = perturb_node_features(x=pert_x, node_idx=n_id, pert_feat=torch.arange(pert_x.shape[1]), bin_dims=sens_idx, device = device)

        org_vec = F.softmax(model(x, edge_index)[n_id], dim=-1)
        org_pred = torch.argmax(org_vec)
        pert_vec = F.softmax(model(pert_x, pert_edge_index)[n_id], dim=-1)
        pert_pred = torch.argmax(pert_vec)

        if org_pred.item() == pert_pred.item():
            if rep == 'softmax':
                # Softmax differences
                L_softmax = torch.dist(org_vec, pert_vec, p=dist_norm)
                delta_softmax.append(L_softmax.item())

            elif rep == 'intermediate':
                raise NotImplementedError('Intermediate model check will be implemented in the future version!')

            elif rep == 'final':
                raise NotImplementedError('Final embedding check will be implemented in the future version!')

            else:
                print('Invalid choice! Exiting..')

    if rep == 'softmax':
        # print(delta_softmax)
        return np.mean(delta_softmax)

    elif rep == 'intermediate':
        raise NotImplementedError('Intermediate model check will be implemented in the future version!')
        # return [np.mean(delta_L1), np.mean(delta_L2)]

    elif rep == 'final':
        raise NotImplementedError('Final embedding check will be implemented in the future version!')
        # return np.mean(delta_Lfinal)

    else:
        print('Invalid choice! Exiting...')
        exit(0)


def check_delta(x, edge_index, model, rep, pert_x, pert_edge_index, n_id, delta, dist_norm=2):
    if rep == 'softmax':
        # Softmax differences
        org_softmax = F.softmax(model(x.to(device), edge_index.to(device))[n_id], dim=-1)
        org_pred = torch.argmax(org_softmax)
        pert_softmax = F.softmax(model(pert_x.to(device), pert_edge_index.to(device))[n_id], dim=-1)
        pert_pred = torch.argmax(pert_softmax)
        return torch.dist(org_softmax, pert_softmax, p=dist_norm).item() <= delta

    elif rep == 'final':
        raise NotImplementedError('Final embedding check will be implemented in the future version!')

    elif rep == 'intermediate':
        raise NotImplementedError('Intermediate model check will be implemented in the future version!')
    else:
        print('Invalid choice! Exiting..')
        exit(0)


def intersection(lst1, lst2):
    return set(lst1).union(lst2)


def graph_exp_stability(generated_exp: Explanation, explainer, 
        shape_graph: ShapeGraph, node_id, model, delta, sens_idx, 
        top_k=0.25, rep='softmax', device = "cpu",
        G = None, data = None, num_run = 25) -> float:
    GES_feat = []
    GES_node = []
    GES_edge = []

    if data is None:
        data = shape_graph.get_graph(use_fixed_split=True)
    X = data.x.to(device)
    Y = data.y.to(device)
    EIDX = data.edge_index.to(device)

    if G is None:
        G = to_networkx_conv(data, to_undirected=True) # Cache graph to use in rewire_edges
    data_for_rewire = Data(edge_index = EIDX, num_nodes = 1)

    for run in range(num_run):
        # Generate perturbed counterpart
        #pert_edge_index = rewire_edges(EIDX, node_idx=node_id, num_nodes=1).to(device)  # , seed=run)
        pert_edge_index = rewire_edges(EIDX, 
            data = data_for_rewire,
            G = G,
            node_idx=node_id, num_nodes=1).to(device)  # , seed=run)
        #try:
            # import time; st_time = time.time()
        #    pert_edge_index = rewire_edges(EIDX, 
                # data = data_for_rewire,
                # G = G,
        #        node_idx=node_id, num_nodes=1).to(device)  # , seed=run)
            # print(time.time()-st_time)
        #except:
        #    continue
        pert_x = X.clone()
        pert_x[node_id] = perturb_node_features(x=pert_x, node_idx=node_id, pert_feat=torch.arange(pert_x.shape[1]), bin_dims=sens_idx, device = device)

        if check_delta(X, EIDX, model.to(device), rep, pert_x.to(device), pert_edge_index.to(device), node_id, delta):
            pert_exp = explainer.get_explanation_node(
                    x=pert_x, 
                    node_idx=node_id, 
                    y = Y.clone(), # MUST COPY
                    edge_index=pert_edge_index)
            # TESTING:
            #print('node', pert_exp.node_imp)
            #print('edge', pert_exp.edge_imp)
            #print('feat', pert_exp.feat_imp)

            if generated_exp.feature_imp is not None:
                top_feat = int(generated_exp.feature_imp.shape[0] * top_k) 
                ori_exp_mask = torch.zeros_like(generated_exp.feature_imp)
                ori_exp_mask[generated_exp.feature_imp.topk(top_feat)[1]] = 1
                pert_exp_mask = torch.zeros_like(pert_exp.feature_imp)
                pert_exp_mask[pert_exp.feature_imp.topk(top_feat)[1]] = 1
                GES_feat.append(1 - F.cosine_similarity(ori_exp_mask.reshape(1, -1), pert_exp_mask.reshape(1, -1)).item())
            if generated_exp.node_imp is not None:
                top_node = int(generated_exp.node_imp.shape[0] * top_k) 

                try:
                    all_nodes = [*intersection([*generated_exp.node_reference], [*pert_exp.node_reference])]
                    ori_exp_mask = torch.zeros([len(all_nodes)])
                    pert_exp_mask = torch.zeros([len(all_nodes)])
                    for i, n_id in enumerate(all_nodes):
                        if n_id in [*generated_exp.node_reference]:
                            ori_exp_mask[i] = generated_exp.node_imp[generated_exp.node_reference[n_id]].item()
                        if n_id in [*pert_exp.node_reference]:
                            pert_exp_mask[i] = pert_exp.node_imp[pert_exp.node_reference[n_id]].item()
                    if (pert_exp.node_imp.unique().shape[0] == 2) and (generated_exp.node_imp.unique().shape[0] == 2):
                        GES_node.append(1 - F.cosine_similarity(ori_exp_mask.reshape(1, -1), pert_exp_mask.reshape(1, -1)).item())
                    else:
                        topk, indices = torch.topk(ori_exp_mask, top_node)
                        ori_exp_mask = torch.zeros_like(ori_exp_mask).scatter_(0, indices, topk)
                        ori_exp_mask[ori_exp_mask.topk(top_node)[1]] = 1
                        topk, indices = torch.topk(pert_exp_mask, top_node)
                        pert_exp_mask = torch.zeros_like(pert_exp_mask).scatter_(0, indices, topk)
                        pert_exp_mask[pert_exp_mask.topk(top_node)[1]] = 1
                        GES_node.append(1 - F.cosine_similarity(ori_exp_mask.reshape(1, -1), pert_exp_mask.reshape(1, -1)).item())
                except:
                    continue
            if generated_exp.edge_imp is not None:
                org_edges = torch.where(generated_exp.enc_subgraph.edge_mask == True)[0]

                # Create a dictionary mapping edge ids to their importance
                org_map = {}
                for (i, edge) in enumerate(org_edges):
                    org_map[edge.item()] = generated_exp.edge_imp[i].item()
                   
                pert_edges = torch.where(pert_exp.enc_subgraph.edge_mask == True)[0].to(device)

                # Create a dictionary mapping edge ids to their importance
                pert_map = {}
                for (i, edge) in enumerate(pert_edges):
                    pert_map[edge.item()] = pert_exp.edge_imp[i].item()

                all_edges = torch.from_numpy(np.union1d(org_edges.cpu().numpy(), pert_edges.cpu().numpy()))
                ori_exp_mask = torch.zeros([len(all_edges)]).to(device)
                pert_exp_mask = torch.zeros([len(all_edges)]).to(device)
                for i, e_id in enumerate(all_edges):
                    if e_id.item() in org_edges:
                        ori_exp_mask[i] = org_map[e_id.item()]
                    if e_id.item() in pert_edges:
                        pert_exp_mask[i] = pert_map[e_id.item()]
                GES_edge.append(1 - F.cosine_similarity(ori_exp_mask.reshape(1, -1), pert_exp_mask.reshape(1, -1)).item())

    return [max(GES_feat) if len(GES_feat)>0 else None, max(GES_node) if len(GES_node)>0 else None, max(GES_edge) if len(GES_edge)>0 else None]

# check_delta(x, edge_index, model, rep, pert_x, pert_edge_index, n_id, delta, dist_norm=2)
def graph_exp_cf_fairness(generated_exp: Explanation, gnnexpr, shape_graph: ShapeGraph, model, node_id, delta, sens_idx, top_k=0.25, rep='softmax', device = 'cpu', data = None) -> float:
    GECF_feat = None
    GECF_node = None
    GECF_edge = None

    if data is None:
        data = shape_graph.get_graph(use_fixed_split=True)
    X = data.x.to(device)
    Y = data.y.to(device)
    EIDX = data.edge_index.to(device)

    pert_x = X.clone()
    pert_x[node_id, sens_idx] = 1 - pert_x[node_id, sens_idx]

    #if check_delta(shape_graph, rep, pert_x, EIDX, node_id, delta):
    if check_delta(X, EIDX, model = model, rep = rep, pert_x = pert_x, pert_edge_index = EIDX, n_id = node_id, delta = delta):

        # Test for GNN Explainers
        #gnnexpr = GNNExplainer(model)
        pert_exp = gnnexpr.get_explanation_node(x=pert_x.to(device), node_idx=int(node_id), edge_index=EIDX.to(device), y = Y.clone().to(device))

        if generated_exp.feature_imp is not None:
            top_feat = int(generated_exp.feature_imp.shape[0] * top_k) 
            ori_exp_mask = torch.zeros_like(generated_exp.feature_imp)
            ori_exp_mask[generated_exp.feature_imp.topk(top_feat)[1]] = 1
            pert_exp_mask = torch.zeros_like(pert_exp.feature_imp)
            pert_exp_mask[pert_exp.feature_imp.topk(top_feat)[1]] = 1
            GECF_feat = 1 - F.cosine_similarity(ori_exp_mask.reshape(1, -1), pert_exp_mask.reshape(1, -1)).item()

        if generated_exp.node_imp is not None:
            top_node = int(generated_exp.node_imp.shape[0] * top_k) 
            all_nodes = [*intersection([*generated_exp.node_reference], [*pert_exp.node_reference])]         
            ori_exp_mask = torch.zeros([len(all_nodes)])
            pert_exp_mask = torch.zeros([len(all_nodes)])

            for i, n_id in enumerate(all_nodes):
                if n_id in [*generated_exp.node_reference]:
                    ori_exp_mask[i] = generated_exp.node_imp[generated_exp.node_reference[n_id]].item()
                if n_id in [*pert_exp.node_reference]:
                    pert_exp_mask[i] = pert_exp.node_imp[pert_exp.node_reference[n_id]].item()
            if (pert_exp.node_imp.unique().shape[0] == 2) and (generated_exp.node_imp.unique().shape[0] == 2):
                GECF_node = 1 - F.cosine_similarity(ori_exp_mask.reshape(1, -1), pert_exp_mask.reshape(1, -1)).item()
            else:
                topk, indices = torch.topk(ori_exp_mask, top_node)
                ori_exp_mask = torch.zeros_like(ori_exp_mask).scatter_(0, indices, topk)
                ori_exp_mask[ori_exp_mask.topk(top_node)[1]] = 1
                topk, indices = torch.topk(pert_exp_mask, top_node)
                pert_exp_mask = torch.zeros_like(pert_exp_mask).scatter_(0, indices, topk)
                pert_exp_mask[pert_exp_mask.topk(top_node)[1]] = 1
                GECF_node = 1 - F.cosine_similarity(ori_exp_mask.reshape(1, -1), pert_exp_mask.reshape(1, -1)).item()

        if generated_exp.edge_imp is not None:
            org_edges = torch.where(generated_exp.enc_subgraph.edge_mask == True)[0]

            # Create a dictionary mapping edge ids to their importance
            org_map = {}
            for (i, edge) in enumerate(org_edges):
                org_map[edge.item()] = generated_exp.edge_imp[i].item()
                    
            pert_edges = torch.where(pert_exp.enc_subgraph.edge_mask == True)[0]

            # Create a dictionary mapping edge ids to their importance
            pert_map = {}
            for (i, edge) in enumerate(pert_edges):
                pert_map[edge.item()] = pert_exp.edge_imp[i].item()

            all_edges = torch.from_numpy(np.union1d(org_edges.cpu().numpy(), pert_edges.cpu().numpy())).to(device)
            ori_exp_mask = torch.zeros([len(all_edges)]).to(device)
            pert_exp_mask = torch.zeros([len(all_edges)]).to(device)
            for i, e_id in enumerate(all_edges):
                if e_id in org_edges:
                    ori_exp_mask[i] = org_map[e_id.item()]
                if e_id in pert_edges:
                    pert_exp_mask[i] = pert_map[e_id.item()]
            GECF_edge = 1 - F.cosine_similarity(ori_exp_mask.reshape(1, -1), pert_exp_mask.reshape(1, -1)).item()

    return [GECF_feat, GECF_node, GECF_edge]


def stat_parity(org, pred, sens):
    idx_s0 = np.array(sens) == 0
    idx_s1 = np.array(sens) == 1

    if (len(np.unique(sens)) == 1):
        return 0
    else:
        parity_1 = abs(sum(org[idx_s0]) / sum(idx_s0) - sum(org[idx_s1])/sum(idx_s1))
        parity_2 = abs(sum(pred[idx_s0]) / sum(idx_s0) - sum(pred[idx_s1])/sum(idx_s1))
        return abs(parity_1-parity_2)

# check_delta(x, edge_index, model, rep, pert_x, pert_edge_index, n_id, delta, dist_norm=2)


def graph_exp_group_fairness(generated_exp: Explanation, shape_graph: ShapeGraph, 
        node_id, model, delta, sens_idx, top_k=0.25, rep='softmax', device = 'cpu',
        G = None, data = None, num_samples = 10) -> float:

    # Generate the predictions
    org_pred = []
    exp_pred_feat = []
    exp_pred_node = []
    exp_pred_edge = []
    sens_feat = []
    sens_node = []
    sens_edge = []
    GGF_feat = None
    GGF_node = None
    GGF_edge = None

    if data is None:
        data = shape_graph.get_graph()
    X = data.x.to(device)
    Y = data.y.to(device)
    EIDX = data.edge_index.to(device)

    if G is None:
        G = to_networkx_conv(data, to_undirected=True) # Cache graph to use in rewire_edges
    data_for_rewire = Data(edge_index = EIDX, num_nodes = 1)

    # Number of neighborhood samples
    num_samples = 10

    for i in range(num_samples):
        if i == 0:
            # predictions for the original features
            pert_x = X.clone()
            pert_edge_index = EIDX
            out_x = torch.argmax(F.softmax(model(pert_x, pert_edge_index), dim=-1)[node_id]).item()
        else:
            # perturb node features for node_id
            pert_x = X.clone()
            pert_x[node_id, :] = perturb_node_features(x=pert_x, node_idx=node_id, pert_feat=torch.arange(pert_x.shape[1]), bin_dims=[sens_idx], device = device)
            #try:
            pert_edge_index = rewire_edges(EIDX, node_idx=node_id.item() if isinstance(node_id, torch.Tensor) else node_id, num_nodes=1, G = G, data = data_for_rewire).to(device)
            # except:
            #     continue

            # Get predictions
            out_x = torch.argmax(F.softmax(model(pert_x, pert_edge_index), dim=-1)[node_id]).item()
       
        #if check_delta(shape_graph, rep, pert_x, pert_edge_index, node_id, delta):
        if check_delta(X, EIDX, model = model, rep = rep, pert_x = pert_x, pert_edge_index = pert_edge_index, n_id = node_id, delta = delta):

            if generated_exp.feature_imp is not None:
                top_feat = int(generated_exp.feature_imp.shape[0] * top_k) 
                ori_exp_mask = torch.zeros_like(generated_exp.feature_imp).to(device)
                ori_exp_mask[generated_exp.feature_imp.topk(top_feat)[1]] = 1
                pert_x_feat = pert_x.clone()
                sens_feat.append(pert_x_feat[node_id, sens_idx].item())
                pert_x_feat[node_id] = pert_x_feat[node_id, :].mul(ori_exp_mask)
                out_exp_x = torch.argmax(F.softmax(model(pert_x_feat, pert_edge_index), dim=-1)[node_id]).item()
                exp_pred_feat.append(out_exp_x)

            if generated_exp.node_imp is not None:
                # Identifying the top_k nodes in the explanation subgraph
                top_k_nodes = generated_exp.node_imp.topk(int(generated_exp.node_imp.shape[0] * top_k))[1] 

                # Get the nodes for removal
                rem_nodes = []
                for node in range(generated_exp.node_imp.shape[0]):
                    if node not in top_k_nodes:
                        rem_nodes.append([k for k, v in generated_exp.node_reference.items() if v == node][0])

                # Getting the softmax vector for the perturbed graph
                # Removing the unimportant nodes by masking
                pert_x_node = pert_x.clone()
                pert_x_node[rem_nodes] = torch.zeros_like(pert_x_node[rem_nodes]).to(device)
                out_exp_x = torch.argmax(F.softmax(model(pert_x_node, pert_edge_index), dim=-1)[node_id]).item()
                sens_node.append(pert_x_node[node_id, sens_idx].item())
                exp_pred_node.append(out_exp_x)

            if generated_exp.edge_imp is not None:
                subgraph_edges = torch.where(generated_exp.enc_subgraph.edge_mask == True)[0]

                # Get the list of all edges that we need to keep
                # keep_edges = [] 
                # for i in range(EIDX.shape[1]):
                #     if i in subgraph_edges and generated_exp.edge_imp[(subgraph_edges == i).nonzero(as_tuple=True)[0]]==0:
                #         continue
                #     else:
                #         keep_edges.append(i)
                top_k_edges = generated_exp.edge_imp.topk(int(generated_exp.edge_imp.shape[0] * top_k))[1]

                subgraph_mask = torch.zeros(subgraph_edges.shape[0]).bool()
                subgraph_mask[top_k_edges] = True

                # Bigger indices:
                to_remove = subgraph_edges[subgraph_mask]
                wholeEIDX_mask = torch.ones(EIDX.shape[1]).bool()
                wholeEIDX_mask[to_remove] = False

                #edge_index = EIDX[:,wholeEIDX_mask]

                # Get new edge_index
                pert_edge_index = pert_edge_index[:, wholeEIDX_mask]

                # Getting the softmax vector for the perturbed graph
                out_exp_x = torch.argmax(F.softmax(model(pert_x, pert_edge_index), dim=-1)[node_id]).item()
                sens_edge.append(pert_x[node_id, sens_idx].item())
                exp_pred_edge.append(out_exp_x)


            org_pred.append(out_x)

    # Calculate Statistical Parity
    if len(sens_feat) > 0:
        GGF_feat = stat_parity(np.array(org_pred), np.array(exp_pred_feat), np.array(sens_feat))
    if len(sens_node) > 0:
        GGF_node = stat_parity(np.array(org_pred), np.array(exp_pred_node), np.array(sens_node))
    if len(sens_edge) > 0:
        GGF_edge = stat_parity(np.array(org_pred), np.array(exp_pred_edge), np.array(sens_edge))
    return [GGF_feat, GGF_node, GGF_edge]

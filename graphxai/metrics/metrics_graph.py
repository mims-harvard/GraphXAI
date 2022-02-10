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
from graphxai.datasets.dataset import GraphDataset
from graphxai.utils import to_networkx_conv, Explanation, distance
from graphxai.utils.perturb import rewire_edges, perturb_node_features


device = "cuda" if torch.cuda.is_available() else "cpu"

def graph_exp_acc_graph(gt_exp: List[Explanation], generated_exp: Explanation, node_thresh_factor = 0.1) -> float:
    '''

    Specifically for graph-level explanation accuracy

    Args:
        gt_exp (Explanation): Ground truth explanation from the dataset.
        generated_exp (Explanation): Explanation output by an explainer.
    '''

    EPS = 1e-09
    JAC_feat = None
    JAC_node = None
    JAC_edge = None

    # Accessing the enclosing subgraph. Will be the same for both explanation.:
    #exp_subgraph = generated_exp.enc_subgraph

    exp_graph = generated_exp.graph

    if generated_exp.feature_imp is not None:
        JAC_feat = []
        thresh_feat = generated_exp.feature_imp.mean()
        for exp in gt_exp:
            TPs = []
            FPs = []
            FNs = []
            if exp.feature_imp is not None:
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

    JAC_feat = max(JAC_feat) if len(JAC_feat) > 0 else None

    if generated_exp.node_imp is not None:
        JAC_node = []
        thresh_node = node_thresh_factor*generated_exp.node_imp.max()
        for exp in gt_exp:
            TPs = []
            FPs = []
            FNs = []
            #relative_positives = (exp.node_imp == 1).nonzero(as_tuple=True)[0]
            true_nodes = (exp.node_imp == 1).nonzero(as_tuple=True)[0]
            #true_nodes = [exp.graph.nodes[i].item() for i in relative_positives]
            #for i, node in enumerate(exp_graph.nodes):
            #print('x shape {} \t node_imp shape {}'.format(exp_graph.x.shape, generated_exp.node_imp.shape))
            for node in range(exp_graph.x.shape[0]):
                # Restore original node numbering
                positive = generated_exp.node_imp[node].item() > thresh_node
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


def graph_exp_faith_graph(generated_exp: Explanation, data: Data, 
        model, sens_idx: List[int]= [], top_k: float = 0.25,
        forward_kwargs = {}) -> float:
    '''
    Args:
        gt_exp (Explanation): Ground truth explanation from the dataset.
        generated_exp (Explanation): Explanation output by an explainer.
        forward_kwargs (dict, optional): Any additional arguments to the forward method
            of model, other than x and edge_index.
    '''

    GEF_feat = None
    GEF_node = None
    GEF_edge = None

    # Accessing the enclosing subgraph. Will be the same for both explanation.:
    #exp_graph = generated_exp.graph

    #data = dataset.get_graph(use_fixed_split=True)
    X = data.x.to(device)
    Y = data.y.to(device)
    EIDX = data.edge_index.to(device)

    # Getting the softmax vector for the original graph
    org_vec = model(X, EIDX, **forward_kwargs)
    org_softmax = F.softmax(org_vec, dim=-1)

    if False: #generated_exp.feature_imp is not None:
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

        # Identifying the top_k nodes in the graph
        top_k_nodes = generated_exp.node_imp.topk(int(generated_exp.node_imp.shape[0] * top_k))[1]

        #rem_nodes = []
        # for node in range(generated_exp.node_imp.shape[0]):
        #     if node not in top_k_nodes:
        #         rem_nodes.append([k for k, v in generated_exp.node_reference.items() if v == node][0])

        # Get all nodes not in 
        rem_nodes = [node for node in range(generated_exp.node_imp.shape[0]) if node in top_k_nodes]

        # Getting the softmax vector for the perturbed graph
        pert_x = X.clone()

        # Removing the unimportant nodes by masking
        pert_x[rem_nodes] = torch.zeros_like(pert_x[rem_nodes]).to(device)
        pert_vec = model(pert_x, EIDX, **forward_kwargs)#[generated_exp.node_idx]
        pert_softmax = F.softmax(pert_vec, dim=-1)
        GEF_node = 1 - torch.exp(-F.kl_div(org_softmax.log(), pert_softmax, None, None, 'sum')).item()

    if generated_exp.edge_imp is not None:
        #positive_edges = torch.where(generated_exp.enc_subgraph.edge_mask == True)[0].to(device)
        # Get the list of all edges that we need to keep
        # keep_edges = [] 
        # # Assumes the edge imp is binary
        # for i in range(EIDX.shape[1]):
        #     if generated_exp.edge_imp[i].item() == 1:
        #         continue
        #     else:
        #         keep_edges.append(i)

        keep_edges = torch.where(generated_exp.edge_imp == 1)[0]

        # Get new edge_index
        edge_index = EIDX[:, keep_edges]
                    
        # Getting the softmax vector for the perturbed graph
        pert_vec = model(X, edge_index.to(device), **forward_kwargs)#[generated_exp.node_idx]
        pert_softmax = F.softmax(pert_vec, dim=-1)        
        GEF_edge = 1 - torch.exp(-F.kl_div(org_softmax.log(), pert_softmax, None, None, 'sum')).item()

    return [GEF_feat, GEF_node, GEF_edge]


import random, sys
from typing import Union, List

from networkx.classes.function import to_undirected
import networkx as nx
import ipdb
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from numpy import ndarray
from torch_geometric.utils import to_networkx

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
from graphxai.utils.perturb import rewire_edges


def graph_exp_acc(gt_exp: Explanation, generated_exp: Explanation, threshold = 0.8) -> float:
    '''
    Args:
        gt_exp (Explanation): Ground truth explanation from the dataset.
        generated_exp (Explanation): Explanation output by an explainer.
    '''

    # TODO: 1) Check that subgraphs match
    #       2) Have to implement the cases where we have multiple ground-truth explanations

    EPS = 1e-09
    thresh = threshold
    relative_positives = (gt_exp.node_imp == 1).nonzero(as_tuple=True)[0]
    true_nodes = [gt_exp.enc_subgraph.nodes[i].item() for i in relative_positives]

    # Accessing the enclosing subgraph. Will be the same for both explanation.:
    exp_subgraph = generated_exp.enc_subgraph

    calc_node_imp = generated_exp.node_imp

    TPs = []
    FPs = []
    FNs = []
    for i, node in enumerate(exp_subgraph.nodes):
        # Restore original node numbering
        positive = calc_node_imp[i].item() > thresh
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
    JAC = TP / (TP + FP + FN + EPS)
    # print(f'TP / (TP+FP+FN) edge score of gnn explainer is {edge_score}')

    return JAC


def graph_exp_faith(generated_exp: Explanation, shape_graph: ShapeGraph, model: torch.nn.Module, top_k=0.25) -> float:
    '''
    Args:
        gt_exp (Explanation): Ground truth explanation from the dataset.
        generated_exp (Explanation): Explanation output by an explainer.
    '''

    # TODO: 1) Implement perturbations for continuous and discrete node attribute features

    # Accessing the enclosing subgraph. Will be the same for both explanation.:
    exp_subgraph = generated_exp.enc_subgraph

    if generated_exp.feature_imp is not None:
        # Identifying the top_k features in the node attribute feature vector
        top_k_features = generated_exp.feature_imp.topk(int(generated_exp.feature_imp.shape[0] * top_k))[1]
        node_map = [k for k, v in generated_exp.node_reference.items() if v == generated_exp.node_idx][0]

        # Getting the softmax vector for the original graph
        org_vec = model(shape_graph.get_graph().x, shape_graph.get_graph().edge_index)[generated_exp.node_idx]
        org_softmax = F.softmax(org_vec, dim=-1)

        # Getting the softmax vector for the perturbed graph
        pert_x = shape_graph.get_graph().x.clone()

        # Perturbing the unimportant node feature indices using gaussian noise
        rem_features = torch.Tensor(
            [i for i in range(shape_graph.get_graph().x.shape[1]) if i not in top_k_features]).long()
        pert_x[node_map, rem_features] = torch.normal(0, 0.1, pert_x[node_map, rem_features].shape)
        pert_vec = model(pert_x, shape_graph.get_graph().edge_index)[generated_exp.node_idx]
        pert_softmax = F.softmax(pert_vec, dim=-1)

    if generated_exp.node_imp is not None:
        # Identifying the top_k nodes in the explanation subgraph
        top_k_nodes = generated_exp.node_imp.topk(int(generated_exp.node_imp.shape[0] * top_k))[1]

        rem_nodes = []
        for node in range(generated_exp.node_imp.shape[0]):
            if node not in top_k_nodes:
                rem_nodes.append([k for k, v in generated_exp.node_reference.items() if v == node][0])

        # Getting the softmax vector for the original graph
        org_vec = model(shape_graph.get_graph().x, shape_graph.get_graph().edge_index)[generated_exp.node_idx]
        org_softmax = F.softmax(org_vec, dim=-1)

        # Getting the softmax vector for the perturbed graph
        pert_x = shape_graph.get_graph().x.clone()

        # Removing the unimportant nodes by masking
        pert_x[rem_nodes] = torch.zeros_like(pert_x[rem_nodes])  # torch.normal(0, 0.1, pert_x[rem_nodes].shape)
        pert_vec = model(pert_x, shape_graph.get_graph().edge_index)

        # check if the graph is disconnected!!
        pert_softmax = F.softmax(pert_vec, dim=-1)

    if generated_exp.edge_imp is not None:
        # Identifying the top_k edges in the explanation subgraph
        edge_imp = torch.Tensor([0.1, 0.4, 0.3, 0.25, 0.9])
        top_k_edges = edge_imp.topk(int(edge_imp.shape[0] * top_k))[1]
        # TODO: After using an explanation method that generates edge-level explanation

    GEF = 1 - torch.exp(-F.kl_div(org_softmax.log(), pert_softmax, None, None, 'sum')).item()

    return GEF


# def calculate_delta(
#         generated_exp: Explanation, 
#         train_set, 
#         model: torch.nn.Module, 
#         rep='final', 
#         dist_norm=2):

#     delta_softmax, delta_L1, delta_L2, delta_Lfinal = [], [], [], []

#     for n_id in train_set:
#         try:
#             pert_edge_index = rewire_edges(generated_exp.graph.edge_index, node_idx=n_id.item(), num_nodes=1)
#         except:
#             continue
#         pert_x = generated_exp.graph.x.clone()
#         pert_x[n_id] += torch.normal(0, 0.1, pert_x[n_id].shape)
#         org_vec = F.softmax(model(generated_exp.graph.x, generated_exp.graph.edge_index)[n_id], dim=-1)
#         org_pred = torch.argmax(org_vec)
#         pert_vec = F.softmax(model(pert_x, pert_edge_index)[n_id], dim=-1)
#         pert_pred = torch.argmax(pert_vec)

#         if org_pred.item() == pert_pred.item():
#             if rep == 'softmax':
#                 # Softmax differences
#                 L_softmax = torch.dist(org_vec, pert_vec, p=dist_norm)
#                 delta_softmax.append(L_softmax.item())
#             elif rep == 'intermediate':
#                 # Layer 1 differences
#                 org_layer1 = F.relu(model.gin1(generated_exp.graph.x, generated_exp.graph.edge_index))
#                 pert_layer1 = F.relu(model.gin1(pert_x, pert_edge_index))
#                 L1 = torch.dist(org_layer1[n_id], pert_layer1[n_id], p=dist_norm)

#                 # Layer 2 differences
#                 org_layer2 = F.relu(model.gin2(org_layer1, generated_exp.graph.edge_index))
#                 pert_layer2 = F.relu(model.gin2(pert_layer1, pert_edge_index))
#                 L2 = torch.dist(org_layer2[n_id], pert_layer2[n_id], p=dist_norm)
#                 delta_L1.append(L1.item())
#                 delta_L2.append(L2.item())
#             elif rep == 'final':
#                 # Layer 3 differences
#                 org_layer3 = model(generated_exp.graph.x, generated_exp.graph.edge_index)
#                 # model.gin3(org_layer2, generated_exp.graph.edge_index)
#                 pert_layer3 = model(pert_x, pert_edge_index)
#                 # model.gin3(pert_layer2, pert_edge_index)
#                 Lfinal = torch.dist(org_layer3[n_id], pert_layer3[n_id], p=dist_norm)
#                 delta_Lfinal.append(Lfinal.item())
#             else:
#                 print('Invalid choice! Exiting..')

#     if rep == 'softmax':
#         return np.mean(delta_softmax)
#     elif rep == 'intermediate':
#         return [np.mean(delta_L1), np.mean(delta_L2)]
#     elif rep == 'final':
#         return np.mean(delta_Lfinal)
#     else:
#         print('Invalid choice! Exiting...')
#         exit(0)

def calculate_delta(shape_graph: ShapeGraph, train_set, model: torch.nn.Module, label, rep='softmax', dist_norm=2):
    delta_softmax, delta_L1, delta_L2, delta_Lfinal = [], [], [], []

    for n_id in train_set[torch.randperm(train_set.size()[0])][:100]:
        try:
            pert_edge_index = rewire_edges(shape_graph.get_graph().edge_index, node_idx=n_id.item(), num_nodes=1)
        except:
            continue
        pert_x = shape_graph.get_graph().x.clone()
        pert_x[n_id] += torch.normal(0, 0.01, pert_x[n_id].shape)
        # ipdb.set_trace()
        org_vec = F.softmax(model(shape_graph.get_graph().x, shape_graph.get_graph().edge_index)[n_id], dim=-1)
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


# def check_delta(
#         generated_exp: Explanation, 
#         rep, 
#         pert_x, 
#         pert_edge_index, 
#         n_id, 
#         delta, 
#         model: torch.nn.Module,
#         dist_norm=2
#     ):

#     if rep == 'softmax':
#         # Softmax differences
#         org_softmax = F.relu(model(generated_exp.graph.x, generated_exp.graph.edge_index))
#         pert_softmax = F.relu(model(pert_x, pert_edge_index))
#         return torch.dist(org_softmax[n_id], pert_softmax[n_id], p=dist_norm).item() <= delta
#     elif rep == 'final':
#         # Final embedding differences
#         org_layer3 = model(generated_exp.graph.x, generated_exp.graph.edge_index)
#         pert_layer3 = model(pert_x, pert_edge_index)
#         return torch.dist(org_layer3[n_id], pert_layer3[n_id], p=dist_norm).item() <= delta
#     # elif rep == 'intermediate':
#     #
#     else:
#         print('Invalid choice! Exiting..')
#         exit(0)

def check_delta(shape_graph: ShapeGraph, model: torch.nn.Module, rep, pert_x, pert_edge_index, n_id, delta, dist_norm=2):
    if rep == 'softmax':
        # Softmax differences
        org_softmax = F.softmax(model(shape_graph.get_graph().x, shape_graph.get_graph().edge_index)[n_id], dim=-1)
        org_pred = torch.argmax(org_softmax)
        pert_softmax = F.softmax(model(pert_x, pert_edge_index)[n_id], dim=-1)
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

def graph_exp_stability(generated_exp: Explanation, shape_graph: ShapeGraph, node_id, model, delta, top_k=0.25, rep='softmax') -> float:
    GES = []
    num_run = 25
    for run in range(num_run):
        # Generate perturbed counterpart
        ipdb.set_trace()
        try:
            pert_edge_index = rewire_edges(shape_graph.get_graph().edge_index, node_idx=node_id.item(), num_nodes=1, seed=run)
        except:
            print('I am here')
            continue
        pert_x = shape_graph.get_graph().x.clone()
        pert_x[node_id] += torch.normal(0, 0.01, pert_x[node_id].shape)
        print(run)
        if check_delta(shape_graph, rep, pert_x, pert_edge_index, node_id, delta):
            # Compute CAM explanation
            preds = model(pert_x, pert_edge_index)
            pred = preds[node_id, :].reshape(-1, 1)
            pred_class = pred.argmax(dim=0).item()
            act = lambda x: torch.argmax(x, dim=1)
            cam = CAM(model, activation=act)
            cam_pert_exp = cam.get_explanation_node(
                pert_x,
                node_idx=int(node_id),
                label=pred_class,
                edge_index=pert_edge_index)

            # Normalize the explanations to 0-1 range:
            cam_pert_exp.node_imp = cam_pert_exp.node_imp / torch.max(cam_pert_exp.node_imp)
            top_feat = int(generated_exp.node_imp.shape[0] * top_k)
            # print(ori_exp_mask.reshape(1, -1).shape, pert_exp_mask.reshape(1, -1).shape)
            try:
                if generated_exp.node_imp.shape == cam_pert_exp.node_imp.shape:
                    ori_exp_mask = torch.zeros_like(generated_exp.node_imp)
                    ori_exp_mask[generated_exp.node_imp.topk(top_feat)[1]] = 1
                    pert_exp_mask = torch.zeros_like(cam_pert_exp.node_imp)
                    pert_exp_mask[cam_pert_exp.node_imp.topk(top_feat)[1]] = 1
                    GES.append(1 - F.cosine_similarity(ori_exp_mask.reshape(1, -1), pert_exp_mask.reshape(1, -1)).item())
                else:
                    all_nodes = [*intersection([*generated_exp.node_reference], [*cam_pert_exp.node_reference])]
                    ori_exp_mask = torch.zeros([len(all_nodes)])
                    pert_exp_mask = torch.zeros([len(all_nodes)])
                    for i, n_id in enumerate(all_nodes):
                        if n_id in [*generated_exp.node_reference]:
                            ori_exp_mask[i] = generated_exp.node_imp[generated_exp.node_reference[n_id]].item()
                        if n_id in [*cam_pert_exp.node_reference]:
                            pert_exp_mask[i] = cam_pert_exp.node_imp[cam_pert_exp.node_reference[n_id]].item()
                    topk, indices = torch.topk(ori_exp_mask, top_feat)
                    ori_exp_mask = torch.zeros_like(ori_exp_mask).scatter_(0, indices, topk)
                    ori_exp_mask[ori_exp_mask.topk(top_feat)[1]] = 1
                    topk, indices = torch.topk(pert_exp_mask, top_feat)
                    pert_exp_mask = torch.zeros_like(pert_exp_mask).scatter_(0, indices, topk)
                    pert_exp_mask[pert_exp_mask.topk(top_feat)[1]] = 1
                    GES.append(1 - F.cosine_similarity(ori_exp_mask.reshape(1, -1), pert_exp_mask.reshape(1, -1)).item())
            except:
                continue
            print(GES)
    return max(GES)

class Metrics:

    def __init__(self, model, explainer):
        self.model = model
        self.explainer = explainer
        pass

    # Metric 1-3: Different metrics implemented within the class

    def metric1(self):
        pass

    def metric2(self):
        pass

    def metric3(self):
        pass

    def evaluate(self, name: str = 'all'):
        '''
        Args:
            name (str): Name of metric to evaluate
        '''
        pass
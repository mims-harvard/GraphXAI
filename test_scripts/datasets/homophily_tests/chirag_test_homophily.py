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
from torch_geometric.utils import to_networkx, convert
from torch_geometric.utils import sort_edge_index, to_undirected

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


def are_neighbors(edge_index, node1, node2):
    '''
    Determine if the nodes are neighbors in the graph
    '''

    edge12 = torch.any((edge_index[0,:] == node1) | (edge_index[1,:] == node2))
    edge21 = torch.any((edge_index[0,:] == node2) | (edge_index[1,:] == node1))

    return (edge12.item() or edge21.item())


def homophily_test(SG: ShapeGraph):

    data = SG.get_graph()

    sim_samelabel = []
    sim_difflabel = []

    eidx = data.edge_index.clone()

    print(nx.is_connected(convert.to_networkx(data, to_undirected=True)))
    eidx = to_undirected(eidx) # Ensure we're completely undirected
    eidx = sort_edge_index(eidx)[0]
    for i in range(eidx.shape[1]):

        e = eidx[:,i]

        if e[0] > e[1]:
            # We know we've already seen this edge
            continue

        sim = F.cosine_similarity(data.x[e[0], :], data.x[e[1],:], dim = 0)

        if data.y[e[0]] == data.y[e[1]]:
            sim_samelabel.append(sim.item())
        else:
            sim_difflabel.append(sim.item())
    ipdb.set_trace()

    # for i in trange(data.x.shape[0]):
    #     for j in range(i + 1, data.x.shape[0]):

    #         # Determine if they're neighbors:
    #         if not are_neighbors(data.edge_index, i, j):
    #             continue

    #         sim = F.cosine_similarity(data.x[i, :], data.x[j,:])

    #         if data.y[i] == data.y[j]:
    #             sim_samelabel.append(sim.item())
    #         else:

if __name__ == '__main__':

    bah = ShapeGraph(model_layers=3, seed=912, make_explanations=True, num_subgraphs=300, prob_connection=0.0075, subgraph_size=12, class_sep=0.5, n_informative=6, verify=True)
   
    same, diff = homophily_test(bah)
    print('Samples in Class 0', torch.sum(bah.get_graph().y == 0).item())
    print('Samples in Class 1', torch.sum(bah.get_graph().y == 1).item())
 
    print('Mean cosine similarity of Same Label:', np.mean(same))
    print('Mean cosine similarity of Different Label:', np.mean(diff))

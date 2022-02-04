import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from graphxai.datasets import ShapeGraph, load_ShapeGraph
from test_homophily import homophily_test

my_data_root = '/Users/owenqueen/Desktop/HMS_research/graphxai_project/GraphXAI/data/ShapeGraph/new_unzipped'

if __name__ == '__main__':

    HF = 1.0

    # SG = ShapeGraph(
    #     model_layers = 3,
    #     make_explanations=False,
    #     num_subgraphs = 1200,
    #     prob_connection = 0.006,
    #     subgraph_size = 11,
    #     class_sep = 0.6,
    #     n_informative = 4,
    #     homophily_coef = HF,
    #     n_clusters_per_class = 2,
    #     seed = 1456,
    #     verify = False,
    #     attribute_sensitive_feature = False,
    #     add_sensitive_feature = True,
    #     #sens_attribution_noise = 0.75,
    # )

    SG = load_ShapeGraph('SG_HF_HF=-1.pickle', root = my_data_root)

    # triangle = nx.Graph()
    # triangle.add_nodes_from([0, 1, 2])
    # triangle.add_edges_from([(0, 1), (1, 2), (2, 0)])

    # SG = ShapeGraph(
    #     model_layers = 3,
    #     shape = triangle, # NEW SHAPE
    #     num_subgraphs = 1300,
    #     prob_connection = 0.006,
    #     subgraph_size = 12,
    #     class_sep = 0.5,
    #     n_informative = 4,
    #     verify = True,
    #     make_explanations = True,
    #     homphily_coef = HF,
    #     seed = 1456,
    #     add_sensitive_feature = True,
    #     attribute_sensitive_feature = False
    # )

    print(SG.n_clusters_per_class)

    con0, ncon0 = homophily_test(SG, label = 0)

    print('Mean cosine similarity of connected nodes (class 0):', np.mean(con0).item())
    print('Mean cosine similarity of disconnected nodes (class 0):', np.mean(ncon0).item())

    plt.boxplot([con0, ncon0])
    plt.xticks(ticks = [1, 2], labels = ['Connected', 'Disconnected'])
    plt.ylabel('Cosine similarity')
    plt.title(f'Similarities of Label 0 Nodes (HF = {HF})')
    plt.show()

    con1, ncon1 = homophily_test(SG, label = 1, k_sample = 500)

    print('Mean cosine similarity of connected nodes (class 1):', np.mean(con1).item())
    print('Mean cosine similarity of disconnected nodes (class 1):', np.mean(ncon1).item())

    plt.boxplot([con1, ncon1])
    plt.xticks(ticks = [1, 2], labels = ['Connected', 'Disconnected'])
    plt.ylabel('Cosine similarity')
    plt.title(f'Similarities of Label 1 Nodes (HF = {HF})')
    plt.show()
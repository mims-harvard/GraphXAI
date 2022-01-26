import torch
import numpy as np
import matplotlib.pyplot as plt

from graphxai.datasets import ShapeGraph
from test_homophily import homophily_test

if __name__ == '__main__':

    HF = 0.00001

    SG = ShapeGraph(
        model_layers = 3,
        num_subgraphs = 100,
        prob_connection = 0.08,
        subgraph_size = 10,
        class_sep = 1,
        n_informative = 6,
        n_clusters_per_class = 1,
        verify = False,
        make_explanations=False,
        homophily_coef = HF
    )

    con0, ncon0 = homophily_test(SG, label = 0)

    print('Mean cosine similarity of connected nodes (class 0):', np.mean(con0).item())
    print('Mean cosine similarity of disconnected nodes (class 0):', np.mean(ncon0).item())

    plt.boxplot([con0, ncon0])
    plt.xticks(ticks = [1, 2], labels = ['Connected', 'Disconnected'])
    plt.ylabel('Cosine similarity')
    plt.title(f'Similarities of Label 0 Nodes (HF = {HF})')
    plt.show()

    con1, ncon1 = homophily_test(SG, label = 1)

    print('Mean cosine similarity of connected nodes (class 1):', np.mean(con1).item())
    print('Mean cosine similarity of disconnected nodes (class 1):', np.mean(ncon1).item())

    plt.boxplot([con1, ncon1])
    plt.xticks(ticks = [1, 2], labels = ['Connected', 'Disconnected'])
    plt.ylabel('Cosine similarity')
    plt.title(f'Similarities of Label 1 Nodes (HF = {HF})')
    plt.show()
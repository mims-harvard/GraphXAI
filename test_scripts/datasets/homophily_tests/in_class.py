import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from test_homophily import homophily_test

from graphxai.datasets import ShapeGraph

def new_homophily_test(sg, batch_size):

    data = sg.get_graph()

    x = data.x
    label = data.y
    edge_index = data.edge_index

    # Get indices for connected nodes having same label
    c_inds = torch.randperm(edge_index.shape[1])[:batch_size]
    c_inds = c_inds[label[edge_index.t()[c_inds][:, 0]] == label[edge_index.t()[c_inds][:, 1]]]

    # Get indices for connected nodes having different label
    nc_inds = torch.randperm(edge_index.shape[1])[:batch_size]
    nc_inds = nc_inds[label[edge_index.t()[nc_inds][:, 0]] != label[edge_index.t()[nc_inds][:, 1]]]

    # Compute similarities for all edges in the c_inds:
    c_cos_sim = F.cosine_similarity(x[edge_index.t()[c_inds][:, 0]], x[edge_index.t()[c_inds][:, 1]])
    nc_cos_sim = F.cosine_similarity(x[edge_index.t()[nc_inds][:, 0]], x[edge_index.t()[nc_inds][:, 1]])

    return c_cos_sim, nc_cos_sim

if __name__ == '__main__':

    HF = 1.0

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

    con, ncon = new_homophily_test(SG, batch_size = 1000)

    print('Mean cosine similarity of same label nodes:', torch.mean(con).item())
    print('Mean cosine similarity of different label nodes:', torch.mean(ncon).item())

    plt.boxplot([con.tolist(), ncon.tolist()])
    plt.xticks(ticks = [1, 2], labels = ['Same label', 'Different label'])
    plt.ylabel('Cosine similarity')
    plt.title(f'Similarities of connected nodes (HF = {HF})')
    plt.show()
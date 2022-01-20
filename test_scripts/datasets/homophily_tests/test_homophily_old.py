import os
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import trange
import matplotlib.pyplot as plt

from torch_geometric.utils import sort_edge_index, to_undirected
from graphxai.datasets import ShapeGraph
from graphxai.datasets.load_synthetic import load_ShapeGraph

def are_neighbors(edge_index, node1, node2):
    '''
    Determine if the nodes are neighbors in the graph
    '''

    edge12 = torch.any((edge_index[0,:] == node1) | (edge_index[1,:] == node2))
    edge21 = torch.any((edge_index[0,:] == node2) | (edge_index[1,:] == node1))

    return (edge12.item() or edge21.item())

def homophily_test(SG: ShapeGraph):

    #F.cosine_similarity()

    data = SG.get_graph()

    sim_samelabel = []
    sim_difflabel = []

    eidx = data.edge_index.clone()
    eidx = to_undirected(eidx) # Ensure we're completely undirected
    eidx = sort_edge_index(eidx)

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

    # for i in trange(data.x.shape[0]):
    #     for j in range(i + 1, data.x.shape[0]):

    #         # Determine if they're neighbors:
    #         if not are_neighbors(data.edge_index, i, j):
    #             continue

    #         sim = F.cosine_similarity(data.x[i, :], data.x[j,:])

    #         if data.y[i] == data.y[j]:
    #             sim_samelabel.append(sim.item())
    #         else:
    #             sim_difflabel.append(sim.item())

    return sim_samelabel, sim_difflabel

if __name__ == '__main__':

    root_data = os.path.join('/Users/owenqueen/Desktop/data', 'ShapeGraph')

    #SG = load_ShapeGraph(number=2, root = root_data)

    seps = np.arange(0.25, 2.25, step = 0.25)

    sames = []
    diffs = []

    for sep in seps:
        SG = ShapeGraph(
            model_layers = 3,
            num_subgraphs = 100,
            prob_connection = 0.08,
            subgraph_size = 10,
            class_sep = sep,
            n_informative = 6,
            n_clusters_per_class = 1,
            verify = False,
            make_explanations=False
        )

        same, diff = homophily_test(SG)

        sames.append(np.mean(same))
        diffs.append(np.mean(diff))

        print('Mean cosine similarity of Same Label:', np.mean(same))
        print('Mean cosine similarity of Different Label:', np.mean(diff))


    plt.plot(seps, sames, c = 'red', label = 'Same label')
    plt.plot(seps, diffs, label = 'Different Label', c = 'blue')

    plt.title('Homophily Tests')
    plt.ylabel('Cosine similarity')
    plt.xlabel('Class Separation')
    plt.legend()
    plt.show()
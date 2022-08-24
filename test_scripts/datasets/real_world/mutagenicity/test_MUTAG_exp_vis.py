import random

import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from torch_geometric.utils import to_networkx
from graphxai.datasets.real_world.mutagenicity import Mutagenicity

import graphxai.utils as gxai_utils

mutag = Mutagenicity(root = '.')

# MUTAG notation: [C, N, O, ...]
atom_map = {0: 'C', 4: 'N', 1: 'O', 3: 'H'}

to_samp = [i for i in range(len(mutag)) if mutag.graphs[i].y == 1]

L = random.sample((to_samp), k = 20)

print('len', len(mutag))

# L = [1425, 1426]

for i in L:
    g, exp = mutag[i]

    for e in exp:
        e.visualize_graph(show = True)

    exp = gxai_utils.aggregate_explanations(exp, node_level = False)

    G, pos = exp.visualize_graph(show = False)
    #G = gxai_utils.to_networkx_conv(exp.graph, to_undirected=True)

    node_label = {}

    for j in range(exp.graph.x.shape[0]):
        ind = exp.graph.x[j,:].nonzero(as_tuple = True)[0].item()
        try:
            to_map = atom_map[ind]
        except KeyError:
            to_map = 'E'
        
        node_label[j] = to_map

    #nx.draw_networkx_labels(G, pos, node_label, font_color = 'white')
    plt.title(f'Mol. {i}, Label = {mutag.graphs[i].y.item()}')
    plt.show()

    # if i > 10:
    #     exit()
print('Class imbalance:')
print('Label==0:', np.sum([mutag.graphs[i].y.item() == 0 for i in range(len(mutag))]))
print('Label==1:', np.sum([mutag.graphs[i].y.item() == 1 for i in range(len(mutag))]))
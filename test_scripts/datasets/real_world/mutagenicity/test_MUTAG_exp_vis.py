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

L = random.sample(list(range(len(mutag))), k = 20)

L = [1425, 1426]

for i in L:
    g, exp = mutag[i]

    exp = gxai_utils.aggregate_explanations(exp, node_level = False)

    G, pos = exp.graph_draw(show = False)
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
    plt.title(f'Mol. {i}')
    plt.show()

    # if i > 10:
    #     exit()

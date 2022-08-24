import random

import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from torch_geometric.utils import to_networkx
from graphxai.datasets.real_world.MUTAG import MUTAG

import graphxai.utils as gxai_utils

mutag = MUTAG(root = '.')

# MUTAG notation: [C, N, O, ...]
atom_map = {0: 'C', 1: 'N', 2: 'O'}


for i in range(len(mutag)): #random.sample(list(range(len(mutag))), k = 5):
    g, exp = mutag[i]

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

    nx.draw_networkx_labels(G, pos, node_label, font_color = 'white')
    plt.title(f'Mol. {i}')
    plt.show()

    # if i > 10:
    #     exit()

import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import trange
from graphxai.datasets import Mutagenicity
from graphxai.utils import aggregate_explanations

dataset = Mutagenicity(root = '../../../explainers/mutagenicity/data', test_debug=True)

# Iterate through dataset, check for null explanations

conf_matrix = torch.zeros((2, 2))

atom_map = {0:'C',
1:	'O',
2:	'Cl',
3:	'H',
4:	'N',
5:	'F',
6:	'Br',
7:	'S',
8:	'P',
9:	'I',
10:	'Na',
11:	'K',
12:	'Li',
13:	'Ca'}

shown = 0

for i in trange(len(dataset.graphs)):
    matches = int(dataset.explanations[i][0].has_match)
    yval = int(dataset.graphs[i].y.item())

    if np.random.rand() < -1:
        for exp in dataset.explanations[i]:
            exp.visualize_graph(show=True)
        print('Value:', yval)
        big_exp = aggregate_explanations(dataset.explanations[i], node_level = False)
        G, pos = big_exp.visualize_graph(show = False)

        node_label = {}
        for j in G.nodes:
            ind = big_exp.graph.x[j,:].nonzero(as_tuple = True)[0].item()
            try:
                to_map = atom_map[ind]
            except KeyError:
                to_map = 'E'
        
            node_label[j] = to_map

        nx.draw_networkx_labels(G, pos, node_label, font_color = 'white')
        plt.title('Label={}'.format(yval))
        plt.show()

        shown += 1

    if shown > 20:
        break

    conf_matrix[matches, yval] += 1

print(conf_matrix)
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

def visualize_mol_explanation(data, node_weights, ax = None, atoms = None, 
            weight_map = False, show = True):
    '''Visualize explanation for predictions on a graph'''
    G = to_networkx(data)

    pos = nx.kamada_kawai_layout(G)

    map = {i:node_weights[i] for i in range(len(G.nodes))} if weight_map \
            else {i:atoms[i] for i in range(len(G.nodes))}

    if ax is None:
        nx.draw(G, pos, node_color = node_weights, 
            node_size = 400, cmap = plt.cm.Blues,
            arrows = False)#, with_labels = True)
        nx.draw_networkx_labels(G, pos, labels = map)

    else:
        nx.draw(G, pos, node_color = node_weights, 
            node_size = 400, cmap = plt.cm.Blues,
            arrows = False, ax = ax)#, with_labels = True)
        nx.draw_networkx_labels(G, pos, labels = map, ax = ax)
    
    if show:
        plt.show()
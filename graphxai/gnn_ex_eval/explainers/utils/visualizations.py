import torch
import numpy as np
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data

def visualize_mol_explanation(data: torch.Tensor, node_weights: list, 
            ax: matplotlib.axes.Axes = None, atoms: list = None, 
            weight_map: bool = False, show: bool = True):
    '''
    Visualize explanation for predictions on a graph
    Args:
        data (torch_geometric.data): data representing the entire graph
        node_weights (list): weights by which to color the nodes in the graph
        ax (matplotlib.Axes, optional): axis on which to plot the visualization. 
            If `None`, visualization is plotted on global figure (similar to plt.plot).
            (default: :obj:`None`)
        atoms (list, optional): List of atoms corresponding to each node. Used for 
            node labels on the visualization. (default: :obj:`None`)
        weight_map (bool, optional): If `True`, shows node weights (literal values
            from `node_weights` argument) as the node labels on visualization. If 
            `False`, atoms are used. (default: :obj:`False`)
        show (bool, optional): If `True`, calls `plt.show()` to display visualization.
            (default: :obj:`True`)  
    '''
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

def get_node_weights_subgraph(node_weights, subgraph_nodes):
    if isinstance(node_weights, torch.Tensor):
        node_weights_new = [node_weights[n.item()] for n in subgraph_nodes]
    else:
        node_weights_new = [node_weights[n] for n in subgraph_nodes]

    return node_weights_new


def get_node_weights_dict(node_weights, subgraph_nodes):
    node_weights_dict = {n.item():node_weights[n.item()] for n in subgraph_nodes}
    return node_weights_dict

def visualize_subgraph_explanation(edge_index: torch.Tensor, node_weights: list, node_idx: int = None, 
    ax: matplotlib.axes.Axes = None, weight_map: bool = False, show: bool = True):
    '''
    Visualize node explanation on a subgraph.
    :note: Only shows the largest connected component of the subgraph.
    Args:
        edge_index (torch.Tensor): edge index of the subgraph for which you wish to plot.
        node_weights (list): weights by which to color the nodes in the graph. Must contain 
            node weights for every node in the graph (not just the subgraph).
        node_idx (bool, optional): Node index for which to highlight in visualization.
            If `None`, no node is highlighted. (default: :obj:`None`)
        ax (matplotlib.Axes, optional): axis on which to plot the visualization. 
            If `None`, visualization is plotted on global figure (similar to plt.plot).
            (default: :obj:`None`)
        weight_map (bool, optional): If `True`, shows node weights (literal values
            from `node_weights` argument) as the node labels on visualization. If 
            `False`, subgraph node indices are used. (default: :obj:`False`)
        show (bool, optional): If `True`, calls `plt.show()` to display visualization.
            (default: :obj:`True`)  
    '''

    # Subgraph nodes via unique nodes in edge_index:
    subgraph_nodes = torch.unique(edge_index)

    data = Data(x=subgraph_nodes.reshape(-1, 1), edge_index=edge_index)
    bigG = to_networkx(data, to_undirected=True, remove_self_loops=True)
    Gcc = max(nx.connected_components(bigG), key=len)
    G = bigG.subgraph(Gcc)

    node_weights_subgraph = get_node_weights_subgraph(node_weights, G.nodes)

    pos = nx.kamada_kawai_layout(G)

    if weight_map:
        map = get_node_weights_dict(node_weights, subgraph_nodes)

    if ax is None:
        if weight_map:
            nx.draw(G, pos, node_color = node_weights_subgraph, 
                node_size = 400, cmap = plt.cm.Blues,
                arrows = False)
            nx.draw_networkx_labels(G, pos, labels = map)
        else:
            nx.draw(G, pos, node_color = node_weights_subgraph, 
                node_size = 400, cmap = plt.cm.Blues,
                arrows = False, with_labels = True)

        if node_idx is not None:
            nx.draw(G.subgraph(node_idx), pos, node_color = 'yellow', 
                node_size = 400)

    else:
        if weight_map:
            nx.draw(G, pos, node_color = node_weights_subgraph, 
                node_size = 400, cmap = plt.cm.Blues,
                arrows = False, ax = ax)
            nx.draw_networkx_labels(G, pos, labels = map, ax = ax)

        else: 
            nx.draw(G, pos, node_color = node_weights_subgraph, 
                node_size = 400, cmap = plt.cm.Blues,
                arrows = False, ax = ax, with_labels = True)

        if node_idx is not None:
            nx.draw(G.subgraph(node_idx), pos, node_color = 'yellow', 
                node_size = 400, ax = ax)
    
    if show:
        plt.show()

def visualize_categorical_graph(data: Data, show: bool = True):
    '''
    Shows a graph structure with nodes colored corresponding to their 
        classification.

    Args:
        data (torch_geometric.data.Data): Data object containing x and edge_index
            attributes. Contains graph information
        show (bool, optional): If `True`, calls `plt.show()` to display visualization.
            (default: :obj:`True`)  
    '''
    datagraph = Data(x=data.x, edge_index=data.edge_index)
    bigG = to_networkx(datagraph, to_undirected=True, remove_self_loops=True)

    pos = nx.kamada_kawai_layout(bigG)
    nx.draw(bigG, pos, node_color = data.y.tolist(),
            node_size = 400, cmap = plt.cm.Blues,
            arrows = False, with_labels = True)

    if show:
        plt.show()
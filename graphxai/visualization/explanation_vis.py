import torch
import numpy as np
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from collections import Iterable
from torch_geometric.utils import to_networkx, remove_self_loops, remove_isolated_nodes
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.data import Data
from typing import Optional

from graphxai.utils import Explanation
from graphxai.utils.nx_conversion import to_networkx_conv

def get_node_weights_dict(node_weights, subgraph_nodes):
    node_weights_dict = {n.item():node_weights[n.item()] for n in subgraph_nodes}
    return node_weights_dict

def visualize_mol_explanation(exp, directed = False, highlight_node_idx = False):
    '''
    Visualize an explanation with Networkx drawing tools
    '''

    assert exp.graph is not None, "exp.graph must be set"

    exp.graph.to_networkx_conv(to_undirected = not directed, remove_self_loops=True)

    if highlight_node_idx and exp.node_idx:
        pass

def visualize_node_explanation(
        exp: Explanation, 
        connected: Optional[bool] = True,
        ax: Optional[matplotlib.axes.Axes] = None,
        weight_map: Optional[bool] = False,
        show: Optional[bool] = False,
        fig: Optional[matplotlib.figure.Figure] = None):
    '''
    Visualize an explanation for a node prediction
    Args:
        exp (Explanation): Explanation to visualize
        connected (bool, optional): If True, only shows largest connected component.
            (:default: :obj:`True`)
        ax (matplotlib.axes.Axes, optional): Target axis for plot. 
            (:default: :obj:`None`)
        weight_map (bool, optional): If True, shows the weights from the explanation
            on each node. (:default: :obj:`False`)
        show (bool, optional): If True, calls `plt.show()`. (:default: :obj:`False`)
        fig (matplotlib.figure.Figure): Target figure if `ax` provided.
            (:default: :obj:`None`)
    
    No return value
    '''

    subgraph_nodes = exp.enc_subgraph.nodes

    #G, node_map = exp.enc_subgraph_to_networkx(to_undirected=True, remove_self_loops=True, get_map=True)
    G, node_map = exp.enc_subgraph_to_networkx(get_map=True)
    rev_map = {v:k for k, v in node_map.items()}

    node_idx = node_map[exp.node_idx]
    print('In visualization', node_idx)

    print(G.nodes)

    if connected: # Enforces that subgraph is connected
        Gcc = max(nx.connected_components(G), key=len)
        G = G.subgraph(Gcc)

    if exp.node_imp is None:
        node_weights_subgraph = '#1f78b4' # Default value for node color
        cmap = None
    else:
        #node_weights_subgraph = get_node_weights_subgraph(node_weights, G.nodes)
        node_weights_subgraph = [G.nodes[i]['node_imp'] for i in G.nodes]
        cmap = plt.cm.Blues

    if exp.edge_imp is None:
        edge_weights_subgraph = 'k'
        edge_cmap = None
    else:
        #edge_weights_subgraph = {i:[edge_attr[i].item()] for i in range(len(G.edges))}
        edge_weights_subgraph = [G.edges[u, v]['edge_imp'] for u, v in G.edges]
        edge_cmap = plt.cm.Reds

    pos = nx.kamada_kawai_layout(G)

    if ax is None:
        if weight_map:
            nx.draw(G, pos, node_color = node_weights_subgraph, 
                node_size = 400, cmap = plt.cm.Blues, 
                edge_color = edge_weights_subgraph, edge_cmap = edge_cmap,
                arrows = False)
            nx.draw_networkx_labels(G, pos, 
                labels = get_node_weights_dict(exp.node_imp, subgraph_nodes))
        else:
            nx.draw(G, pos, node_color = node_weights_subgraph, 
                node_size = 400, cmap = cmap,
                edge_color = edge_weights_subgraph, edge_cmap = edge_cmap,
                arrows = False)#, with_labels = True)
            nx.draw_networkx_labels(G, pos, labels = {n:rev_map[n] for n in G.nodes})

        if node_idx is not None:
            nx.draw(G.subgraph(node_idx), pos, node_color = 'yellow', 
                node_size = 400)

        if exp.edge_imp is not None:
            sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=min(exp.edge_imp.tolist()), 
                vmax=max(exp.edge_imp.tolist())))
            plt.colorbar(sm, shrink=0.75)

    else:
        if weight_map:
            nx.draw(G, pos, node_color = node_weights_subgraph, 
                node_size = 400, cmap = cmap,
                edge_color = edge_weights_subgraph, edge_cmap = edge_cmap,
                arrows = False, ax = ax)
            nx.draw_networkx_labels(G, pos, 
                labels = get_node_weights_dict(exp.node_imp, subgraph_nodes), ax = ax)

        else: 
            nx.draw(G, pos, node_color = node_weights_subgraph, 
                node_size = 400, cmap = cmap,
                edge_color = edge_weights_subgraph, edge_cmap = edge_cmap,
                arrows = False, ax = ax)#, with_labels = True)
            nx.draw_networkx_labels(G, pos, labels = {n:rev_map[n] for n in G.nodes}, ax = ax)

        if node_idx is not None:
            nx.draw(G.subgraph(node_idx), pos, node_color = 'yellow', 
                node_size = 400, ax = ax)

        if (exp.edge_imp is not None) and (fig is not None):
            sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=min(exp.edge_imp.tolist()), 
                vmax=max(exp.edge_imp.tolist())))
            fig.colorbar(sm, shrink=0.75, ax = ax)
    
    if show:
        plt.show()

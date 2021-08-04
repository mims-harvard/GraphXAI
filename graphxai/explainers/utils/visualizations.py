import torch
import numpy as np
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from collections import Iterable
from torch_geometric.utils import to_networkx, remove_self_loops, remove_isolated_nodes
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.data import Data

# For DIG function:
from torch_geometric.utils import k_hop_subgraph, add_self_loops
from math import sqrt

from . import to_networkx_conv

def visualize_mol_explanation(data: torch.Tensor, node_weights: list = None, 
            edge_weights: list = None,
            ax: matplotlib.axes.Axes = None, atoms: list = None, 
            weight_map: bool = False, show: bool = True,
            directed: bool = False, fig: matplotlib.figure.Figure = None):
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
        directed (bool, optional): If `True`, shows molecule as directed graph.
            (default: :obj:`False`) 
        fig (matplotlib.figure.Figure, optional): Figure for plots being drawn with this
            function. Will be used to direct the colorbar. (default: :obj:`None`)
    '''
    if directed:
        G = to_networkx_conv(data, to_undirected=False, remove_self_loops=True)
    else:
        G = to_networkx_conv(data, to_undirected=True, remove_self_loops=True)

    pos = nx.kamada_kawai_layout(G)

    if node_weights is None:
        node_weights = '#1f78b4'
        map = {i:atoms[i] for i in range(len(G.nodes))}
    else:
        map = {i:node_weights[i] for i in range(len(G.nodes))} if weight_map \
                else {i:atoms[i] for i in range(len(G.nodes))}

    edge_cmap = None
    edge_map = 'k'
    if edge_weights is not None:
        edge_map = {i:[edge_weights[i]] for i in range(len(G.edges))}
        edge_cmap = plt.cm.Reds

    if ax is None:
        nx.draw(G, pos, node_color = node_weights, 
            node_size = 400, cmap = plt.cm.Blues,
            arrows = False, edge_color = edge_map,
            edge_cmap = edge_cmap)
        nodes = nx.draw_networkx_nodes(G, pos, node_size=400, 
            node_color = node_weights, cmap = plt.cm.Blues)

        # Set up colormap:
        if node_weights != '#1f78b4':
            sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=min(node_weights), vmax=max(node_weights)))
            plt.colorbar(sm, shrink=0.75)
        elif edge_weights is not None:
            sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=min(edge_weights), vmax=max(edge_weights)))
            plt.colorbar(sm, shrink=0.75)

        nx.draw_networkx_labels(G, pos, labels = map)

    else:
        nx.draw(G, pos, node_color = node_weights, 
            node_size = 400, cmap = plt.cm.Blues,
            edge_color = edge_map,
            arrows = False, edge_cmap = edge_cmap, ax = ax)#, with_labels = True)
        nx.draw_networkx_labels(G, pos, labels = map, ax = ax)

        if node_weights != '#1f78b4' and (fig is not None):
            sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=min(node_weights), vmax=max(node_weights)))
            fig.colorbar(sm, shrink=0.75, ax=ax)

        if (edge_weights is not None) and (fig is not None):
            sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=min(edge_weights), 
                vmax=max(edge_weights)))
            fig.colorbar(sm, shrink=0.75, ax = ax)
    
    if show:
        plt.show()

def get_node_weights_subgraph(node_weights, subgraph_nodes):
    if isinstance(node_weights, torch.Tensor):
        #node_weights_new = [node_weights[n.item()] for n in subgraph_nodes]
        node_weights_new = [node_weights[n] for n in subgraph_nodes]
    else:
        node_weights_new = [node_weights[n] for n in subgraph_nodes]

    return node_weights_new

def get_edge_weights_subgraph(edge_index, edge_attr):
    map = {}
    for i in range(edge_index.shape[1]):
        map[tuple(edge_index[:,i].tolist())] = edge_attr[i].item()
    return map

def get_node_weights_dict(node_weights, subgraph_nodes):
    node_weights_dict = {n.item():node_weights[n.item()] for n in subgraph_nodes}
    return node_weights_dict

def visualize_subgraph_explanation(edge_index: torch.Tensor, node_weights: list = None, 
    edge_weights: list = None, node_idx: int = None, ax: matplotlib.axes.Axes = None, 
    weight_map: bool = False, show: bool = True, connected: bool = True, 
    fig: matplotlib.figure.Figure = None):
    '''
    Visualize node explanation on a subgraph.
    :note: Only shows the largest connected component of the subgraph.
    Args:
        edge_index (torch.Tensor): edge index of the subgraph for which you wish to plot.
        node_weights (list size(n,), optional): Weights by which to color the nodes in the graph.
            Must be of size (n,), where n is the number of nodes in the subgraph. 
            Contains node weights for every node in the subgraph, ordered by sorted ordering
            of nodes in edge_index.If `None`, all nodes are given same color in drawing. 
            (default: :obj:`None`)
        edge_weights (list size(e,), optional): Weights by which to color the edges in the graph.
            Must be of size (e,), where e is the number of nodes in the subgraph.
            Contains edge_weights for every node in the subgraph, ordered with respect to the
            edges in `edge_index`. (default: :obj:`None`)
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
        connected (bool, optional): If `True`, forces the drawn subgraph to be connected.
            (default: :obj:`True`)
        fig (matplotlib.figure.Figure, optional): Provided if one desires a colorbar for the 
            node or edge coloring in the ax option of the function. (default: :obj:`True`)
    '''

    if edge_weights is None:
        edge_index, _ = remove_self_loops(edge_index)
    else:
        edge_index, edge_attr = remove_self_loops(edge_index, torch.tensor(edge_weights))
        
    # Subgraph nodes via unique nodes in edge_index:
    subgraph_nodes = torch.unique(edge_index)

    data = Data(x=subgraph_nodes.reshape(-1, 1), edge_index=edge_index)
    #G = to_networkx(data, to_undirected=True, remove_self_loops=True)
    G, node_map = to_networkx_conv(data, to_undirected=True, remove_self_loops=True, get_map = True)
    rev_map = {v:k for k, v in node_map.items()}

    node_idx = node_map[node_idx] if node_idx is not None else None

    if connected:
        Gcc = max(nx.connected_components(G), key=len)
        G = G.subgraph(Gcc)

    if node_weights is None:
        node_weights_subgraph = '#1f78b4' # Default value for node color
        cmap = None
    else:
        node_weights_subgraph = get_node_weights_subgraph(node_weights, G.nodes)
        cmap = plt.cm.Blues

    if edge_weights is None:
        edge_weights_subgraph = 'k'
        edge_cmap = None
    else:
        edge_weights_subgraph = {i:[edge_attr[i].item()] for i in range(len(G.edges))}
        edge_cmap = plt.cm.Reds

    pos = nx.kamada_kawai_layout(G)

    if ax is None:
        if weight_map:
            nx.draw(G, pos, node_color = node_weights_subgraph, 
                node_size = 400, cmap = plt.cm.Blues, 
                edge_color = edge_weights_subgraph, edge_cmap = edge_cmap,
                arrows = False)
            nx.draw_networkx_labels(G, pos, 
                labels = get_node_weights_dict(node_weights, subgraph_nodes))
        else:
            nx.draw(G, pos, node_color = node_weights_subgraph, 
                node_size = 400, cmap = cmap,
                edge_color = edge_weights_subgraph, edge_cmap = edge_cmap,
                arrows = False)#, with_labels = True)
            nx.draw_networkx_labels(G, pos, labels = {n:rev_map[n] for n in G.nodes})

        if node_idx is not None:
            nx.draw(G.subgraph(node_idx), pos, node_color = 'yellow', 
                node_size = 400)

        if edge_weights is not None:
            sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=min(edge_attr.tolist()), 
                vmax=max(edge_attr.tolist())))
            plt.colorbar(sm, shrink=0.75)

    else:
        if weight_map:
            nx.draw(G, pos, node_color = node_weights_subgraph, 
                node_size = 400, cmap = cmap,
                edge_color = edge_weights_subgraph, edge_cmap = edge_cmap,
                arrows = False, ax = ax)
            nx.draw_networkx_labels(G, pos, 
                labels = get_node_weights_dict(node_weights, subgraph_nodes), ax = ax)

        else: 
            nx.draw(G, pos, node_color = node_weights_subgraph, 
                node_size = 400, cmap = cmap,
                edge_color = edge_weights_subgraph, edge_cmap = edge_cmap,
                arrows = False, ax = ax)#, with_labels = True)
            nx.draw_networkx_labels(G, pos, labels = {n:rev_map[n] for n in G.nodes}, ax = ax)

        if node_idx is not None:
            nx.draw(G.subgraph(node_idx), pos, node_color = 'yellow', 
                node_size = 400, ax = ax)

        if (edge_weights is not None) and (fig is not None):
            sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=min(edge_attr.tolist()), 
                vmax=max(edge_attr.tolist())))
            fig.colorbar(sm, shrink=0.75, ax = ax)
    
    if show:
        plt.show()

def visualize_subgraph_explanation_w_edge(subgraph_nodes, subgraph_edge_index: torch.Tensor, 
    edge_weights: list = None, node_idx: int = None, 
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

    # Trim data to remove self-loops and isolated nodes:
    edge_index, edge_attr = remove_self_loops(subgraph_edge_index, torch.tensor(edge_weights))

    # Instantiate data object
    data = Data(x=subgraph_nodes.reshape(-1, 1), edge_index=edge_index, edge_attr=edge_attr)

    G = to_networkx(data, to_undirected=True, remove_self_loops=True)
    Gcc = max(nx.connected_components(G), key=len)
    G = G.subgraph(Gcc)

    edge_weights_subgraph = {i:[edge_attr[i].item()] for i in range(len(G.edges))}

    pos = nx.kamada_kawai_layout(G)

    # if weight_map:
    #     map = get_node_weights_dict(node_weights, subgraph_nodes)

    if ax is None:
        if weight_map:
            nx.draw(G, pos, node_color = 'black', 
                node_size = 400,
                arrows = False, edge_color = edge_weights_subgraph,
                # edge_vmin = min(edge_weights_subgraph),
                # edge_vmax = max(edge_weights_subgraph),
                edge_cmap = plt.cm.Reds)
            nx.draw_networkx_labels(G, pos, labels = map)
        else:
            nx.draw(G, pos, node_color = 'black', 
                node_size = 400, cmap = plt.cm.Blues,
                arrows = False, with_labels = True, 
                edge_color = edge_weights_subgraph,
                edge_cmap = plt.cm.Reds)

        if node_idx is not None:
            nx.draw(G.subgraph(node_idx), pos, node_color = 'yellow', 
                node_size = 400)

        sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=min(edge_attr.tolist()), vmax=max(edge_attr.tolist())))
        plt.colorbar(sm, shrink=0.75)

    else:
        if weight_map:
            nx.draw(G, pos, node_color = 'black', 
                node_size = 400, cmap = plt.cm.Blues,
                arrows = False, ax = ax)
            nx.draw_networkx_labels(G, pos, labels = map, ax = ax)

        else: 
            nx.draw(G, pos, node_color = 'black', 
                node_size = 400, cmap = plt.cm.Blues,
                arrows = False, ax = ax, with_labels = True)

        if node_idx is not None:
            nx.draw(G.subgraph(node_idx), pos, node_color = 'yellow', 
                node_size = 400, ax = ax)
    
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=min(edge_attr.tolist()), vmax=max(edge_attr.tolist())))
        ax.colorbar(sm, shrink=0.75)

    if show:
        plt.show()

def visualize_categorical_graph(data: Data = None, nx_graph: nx.classes.graph.Graph = None, node_idx: int = None,
    show: bool = True):
    '''
    Shows a graph structure with nodes colored corresponding to their 
        classification.

    .. note::
        Either data or nx_graph must be specified, but not both. If by some error, both are specified as
        arguments, data argument will be preferred.

    Args:
        data (torch_geometric.data.Data, optional): Data object containing x and edge_index
            attributes. Contains graph information. (default: :obj:`None`)
        nx_graph (nx.classes.graph.Graph, optional): Networkx graph to draw. (default: :obj:`None`)
        show (bool, optional): If `True`, calls `plt.show()` to display visualization.
            (default: :obj:`True`)  
    '''
    assert (data is not None) or (nx_graph is not None), 'Either data or nx_graph args must not be none.'

    if nx_graph is None:
        datagraph = Data(x=data.x, edge_index=data.edge_index)
        bigG = to_networkx(datagraph, to_undirected=True, remove_self_loops=True)
    else:
        bigG = nx_graph

    pos = nx.kamada_kawai_layout(bigG)
    nx.draw(bigG, pos, node_color = data.y.tolist(),
            node_size = 400, cmap = plt.cm.Blues,
            arrows = False, with_labels = True)

    if node_idx is not None:
        nx.draw(bigG.subgraph(node_idx), pos, node_color = 'yellow', 
                node_size = 400)

    if show:
        plt.show()

# sum_edges = lambda x: np.sum(x)

def all_edges_w_node(edge_index, node_idx):
    '''
    Finds all edges containing the given node index
    Controls for self-loops
    Returns indices of the edges containing the node
    '''
    inds_0row = set(all_incoming_edges_w_node(edge_index, node_idx, row = 0))
    inds_1row = set(all_incoming_edges_w_node(edge_index, node_idx, row = 1))
    return list(inds_0row.union(inds_1row))

def all_incoming_edges_w_node(edge_index, node_idx, row = 0):
    '''Gets all incoming edges to a given node provided row index'''
    return (edge_index[row,:] == node_idx).nonzero(as_tuple=True)[0].tolist()

def parse_GNNLRP_explanations(ret_tuple, edge_index, label_idx, 
    edge_agg = np.sum, node_agg = lambda x: np.sum(x)):

    walks, edge_masks, related_predictions = ret_tuple # Unpack

    walk_ids = walks['ids']
    walk_scores = walks['score']

    unique_edges = walk_ids.unique().tolist()
    edge_maps = [[] if e in unique_edges else [0] for e in range(edge_masks[0].shape[0])]
    #edge_maps = list(np.zeros(edge_masks[0].shape[0]))
    #edge_maps = {w:[] for w in walk_ids.unique().tolist()}
    # Use list to preserve order

    for i in range(walk_ids.shape[0]): # Over all walks:
        walk = walk_ids[i,:]
        score = walk_scores[i,label_idx].item()
        walk_nodes = walk.unique().tolist()
        for wn in walk_nodes:
            edge_maps[wn].append(score)

    N = maybe_num_nodes(edge_index)
    node_map = [[] for i in range(N)]

    # Aggregate all edge scores:
    # edge_scores = np.zeros(edge_masks[0].shape[0])
    # for w, x in edge_maps.items():
    #     edge_scores[w] = edge_agg(x)
    edge_scores = [edge_agg(x) for x in edge_maps]
    #edge_scores = list(edge_scores)

    # Iterate over edges, put into their respective portions:
    # Combines edges in given form:
    for n in range(N):
        edge_inds = all_incoming_edges_w_node(edge_index, n)
        for e in edge_inds:
            if isinstance(edge_maps[e], Iterable):
                node_map[n].append(edge_scores[e]) # Append edge scores
            else:
                node_map[n].append([edge_scores[e]])

    # Now combine all incoming edge scores for nodes:
    node_scores = [sum([abs(xi) for xi in x]) for x in node_map]
    print('len node_scores', len(node_scores))

    return node_scores, edge_scores


def visualize_graph(self, node_idx: int, edge_index: torch.Tensor, edge_mask: torch.Tensor, y: torch.Tensor = None,
                        threshold: float = None, nolabel: bool = True, **kwargs): #-> Tuple[Axes, nx.DiGraph]:
    r"""
    Code adapted from Dive into Graphs (DIG)
    Code: https://github.com/divelab/DIG

    Visualizes the subgraph around :attr:`node_idx` given an edge mask
    :attr:`edge_mask`.
    Args:
        node_idx (int): The node id to explain.
        edge_index (LongTensor): The edge indices.
        edge_mask (Tensor): The edge mask.
        y (Tensor, optional): The ground-truth node-prediction labels used
            as node colorings. (default: :obj:`None`)
        threshold (float, optional): Sets a threshold for visualizing
            important edges. If set to :obj:`None`, will visualize all
            edges with transparancy indicating the importance of edges.
            (default: :obj:`None`)
        **kwargs (optional): Additional arguments passed to
            :func:`nx.draw`.
    :rtype: :class:`matplotlib.axes.Axes`, :class:`networkx.DiGraph`
    """
    edge_index, _ = add_self_loops(edge_index, num_nodes=kwargs.get('num_nodes'))
    assert edge_mask.size(0) == edge_index.size(1)

    if self.molecule:
        atomic_num = torch.clone(y)

    # Only operate on a k-hop subgraph around `node_idx`.
    subset, edge_index, _, hard_edge_mask = k_hop_subgraph(
        node_idx, self.__num_hops__, edge_index, relabel_nodes=True,
        num_nodes=None, flow=self.__flow__())

    edge_mask = edge_mask[hard_edge_mask]

    # --- temp ---
    edge_mask[edge_mask == float('inf')] = 1
    edge_mask[edge_mask == - float('inf')] = 0
    # ---

    if threshold is not None:
        edge_mask = (edge_mask >= threshold).to(torch.float)

    if kwargs.get('dataset_name') == 'ba_lrp':
        y = torch.zeros(edge_index.max().item() + 1,
                        device=edge_index.device)
    if y is None:
        y = torch.zeros(edge_index.max().item() + 1,
                        device=edge_index.device)
    else:
        y = y[subset]

    if self.molecule:
        atom_colors = {6: '#8c69c5', 7: '#71bcf0', 8: '#aef5f1', 9: '#bdc499', 15: '#c22f72', 16: '#f3ea19',
                        17: '#bdc499', 35: '#cc7161'}
        node_colors = [None for _ in range(y.shape[0])]
        for y_idx in range(y.shape[0]):
            node_colors[y_idx] = atom_colors[y[y_idx].int().tolist()]
    else:
        atom_colors = {0: '#8c69c5', 1: '#c56973', 2: '#a1c569', 3: '#69c5ba'}
        node_colors = [None for _ in range(y.shape[0])]
        for y_idx in range(y.shape[0]):
            node_colors[y_idx] = atom_colors[y[y_idx].int().tolist()]


    data = Data(edge_index=edge_index, att=edge_mask, y=y,
                num_nodes=y.size(0)).to('cpu')
    G = to_networkx(data, node_attrs=['y'], edge_attrs=['att'])
    mapping = {k: i for k, i in enumerate(subset.tolist())}
    G = nx.relabel_nodes(G, mapping)

    kwargs['with_labels'] = kwargs.get('with_labels') or True
    kwargs['font_size'] = kwargs.get('font_size') or 10
    kwargs['node_size'] = kwargs.get('node_size') or 250
    kwargs['cmap'] = kwargs.get('cmap') or 'cool'

    # calculate Graph positions
    pos = nx.kamada_kawai_layout(G)
    ax = plt.gca()

    for source, target, data in G.edges(data=True):
        ax.annotate(
            '', xy=pos[target], xycoords='data', xytext=pos[source],
            textcoords='data', arrowprops=dict(
                arrowstyle="->",
                lw=max(data['att'], 0.5) * 2,
                alpha=max(data['att'], 0.4),  # alpha control transparency
                color='#e1442a',  # color control color
                shrinkA=sqrt(kwargs['node_size']) / 2.0,
                shrinkB=sqrt(kwargs['node_size']) / 2.0,
                connectionstyle="arc3,rad=0.08",  # rad control angle
            ))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, **kwargs)
    # define node labels
    if self.molecule:
        if nolabel:
            node_labels = {n: f'{self.table(atomic_num[n].int().item())}'
                            for n in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels=node_labels, **kwargs)
        else:
            node_labels = {n: f'{n}:{self.table(atomic_num[n].int().item())}'
                            for n in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels=node_labels, **kwargs)
    else:
        if not nolabel:
            nx.draw_networkx_labels(G, pos, **kwargs)

    return ax, G
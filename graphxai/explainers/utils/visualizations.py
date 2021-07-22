import torch
import numpy as np
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from collections import Iterable
from torch_geometric.utils import to_networkx, remove_self_loops, remove_isolated_nodes
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.data import Data

def visualize_mol_explanation(data: torch.Tensor, node_weights: list = None, 
            edge_weights: list = None,
            ax: matplotlib.axes.Axes = None, atoms: list = None, 
            weight_map: bool = False, show: bool = True,
            directed: bool = False, fig = None):
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
        fig (matplotlib.Figure, optional): Figure for plots being drawn with this
            function. Will be used to direct the colorbar. (default: :obj:`None`)
    '''
    if directed:
        G = to_networkx(data, to_undirected=False, remove_self_loops=True)
    else:
        G = to_networkx(data, to_undirected=True, remove_self_loops=True)
    # print(G.edges)
    # print('Dim of edge_index', data.edge_index.shape)
    # print('Len of networkx edges', len(G.edges))
    # print([G.get_edge_data(*e) for e in G.edges])

    pos = nx.kamada_kawai_layout(G)

    if node_weights is None:
        node_weights = '#1f78b4'
        map = {i:atoms[i] for i in range(len(G.nodes))}
    else:
        map = {i:node_weights[i] for i in range(len(G.nodes))} if weight_map \
                else {i:atoms[i] for i in range(len(G.nodes))}

    edge_cmap = None
    if edge_weights is not None:
        edge_map = {i:[edge_weights[i]] for i in range(len(G.edges))}
        edge_cmap = plt.cm.Reds

    if ax is None:
        nx.draw(G, pos, node_color = node_weights, 
            node_size = 400, cmap = plt.cm.Blues,
            arrows = False, edge_color = edge_map,
            edge_cmap = edge_cmap)#, with_labels = True)
        nodes = nx.draw_networkx_nodes(G, pos, node_size=400, 
            node_color = node_weights, cmap = plt.cm.Blues)
        # pc = matplotlib.collections.PatchCollection(nodes, cmap=plt.cm.Blues)
        # pc.set_array(node_weights)
        # plt.colorbar(pc)

        # Set up colormap:
        if node_weights != '#1f78b4':
            sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=min(node_weights), vmax=max(node_weights)))
            plt.colorbar(sm, shrink=0.75)

        nx.draw_networkx_labels(G, pos, labels = map)

    else:
        nx.draw(G, pos, node_color = node_weights, 
            node_size = 400, cmap = plt.cm.Blues,
            arrows = False, edge_cmap = edge_cmap, ax = ax)#, with_labels = True)
        nx.draw_networkx_labels(G, pos, labels = map, ax = ax)

        # if node_weights != '#1f78b4' and fig is not None:
        #     sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=min(node_weights), vmax=max(node_weights)))
        #     fig.colorbar(sm, shrink=0.75, cax=ax)
    
    if show:
        plt.show()

def get_node_weights_subgraph(node_weights, subgraph_nodes):
    if isinstance(node_weights, torch.Tensor):
        node_weights_new = [node_weights[n.item()] for n in subgraph_nodes]
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

    print('Pre edited edge_index', subgraph_edge_index)

    # Trim data to remove self-loops and isolated nodes:
    edge_index, edge_attr = remove_self_loops(subgraph_edge_index, torch.tensor(edge_weights))
    # print('After rm self loops', edge_index)
    #edge_index, edge_attr, mask = remove_isolated_nodes(edge_index, edge_attr=edge_attr)

    # Subgraph nodes via unique nodes in edge_index:
    #subgraph_nodes = torch.unique(edge_index)

    # print('Subgraph nodes', subgraph_nodes)
    # print('edge_index', edge_index)

    # Instantiate data object
    data = Data(x=subgraph_nodes.reshape(-1, 1), edge_index=edge_index, edge_attr=edge_attr)

    G = to_networkx(data, to_undirected=True, remove_self_loops=True)
    Gcc = max(nx.connected_components(G), key=len)
    G = G.subgraph(Gcc)

    edge_weights_subgraph = {i:[edge_attr[i].item()] for i in range(len(G.edges))}

    print('G nodes', G.nodes)

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

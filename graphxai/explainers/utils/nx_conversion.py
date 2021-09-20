import torch
import numpy as np
import networkx as nx

def to_networkx_conv(data, node_attrs=None, edge_attrs=None, to_undirected=False,
                remove_self_loops=False, get_map = False):
    r"""Converts a :class:`torch_geometric.data.Data` instance to a
    :obj:`networkx.Graph` if :attr:`to_undirected` is set to :obj:`True`, or
    a directed :obj:`networkx.DiGraph` otherwise.

    Args:
        data (torch_geometric.data.Data): The data object.
        node_attrs (iterable of str, optional): The node attributes to be
            copied. (default: :obj:`None`)
        edge_attrs (iterable of str, optional): The edge attributes to be
            copied. (default: :obj:`None`)
        to_undirected (bool, optional): If set to :obj:`True`, will return a
            a :obj:`networkx.Graph` instead of a :obj:`networkx.DiGraph`. The
            undirected graph will correspond to the upper triangle of the
            corresponding adjacency matrix. (default: :obj:`False`)
        remove_self_loops (bool, optional): If set to :obj:`True`, will not
            include self loops in the resulting graph. (default: :obj:`False`)
        get_map (bool, optional): If `True`, returns a tuple where the second
            element is a map from original node indices to new ones.
            (default: :obj:`False`)
    """
    if to_undirected:
        G = nx.Graph()
    else:
        G = nx.DiGraph()

    node_list = sorted(torch.unique(data.edge_index).tolist())
    map_norm = {node_list[i]:i for i in range(len(node_list))}
    #rev_map_norm = {v:k for k, v in map_norm.items()}
    G.add_nodes_from([map_norm[n] for n in node_list])

    values = {}
    for key, item in data:
        if torch.is_tensor(item):
            values[key] = item.squeeze().tolist()
        else:
            values[key] = item
        if isinstance(values[key], (list, tuple)) and len(values[key]) == 1:
            values[key] = item[0]

    for i, (u, v) in enumerate(data.edge_index.t().tolist()):
        u = map_norm[u]
        v = map_norm[v]

        if to_undirected and v > u:
            continue

        if remove_self_loops and u == v:
            continue

        G.add_edge(u, v)
        for key in edge_attrs if edge_attrs is not None else []:
            G[u][v][key] = values[key][i]

    for key in node_attrs if node_attrs is not None else []:
        for i, feat_dict in G.nodes(data=True):
            feat_dict.update({key: values[key][i]})

    if get_map:
        return G, map_norm
    else:
        return G

def mask_graph(edge_index, node_mask = None, edge_mask = None):
    '''
    Masks the edge_index of a graph given either node_mask or edge_mask
    Args:
        edge_index (torch.tensor, dtype=torch.int)
        node_mask (torch.tensor, dtype=bool)
        edge_mask (torch.tensor, dtype=bool)
    '''
    # edge_index's are always size (2,e) with e=number of edges
    if node_mask is not None:
        nodes = node_mask.nonzero(as_tuple=True)[0].tolist()
        created_edge_mask = torch.zeros(edge_index.shape[1])
        for i in range(edge_index.shape[1]):
            edge = edge_index[:,i]
            if (edge[0] in nodes) or (edge[1] in nodes):
                created_edge_mask[i] = 1

        created_edge_mask = created_edge_mask.type(bool)
        edge_index = edge_index[:,created_edge_mask]

    elif edge_mask is not None:
        edge_index = edge_index[:,edge_mask]

    return edge_index

def whole_graph_mask_to_subgraph(node_mask, edge_mask = None, subgraph_nodes = None, subgraph_eidx = None):
    '''Converts mask of whole graph to a mask of a subgraph'''
    nodes = node_mask.nonzero(as_tuple=True)[0]
    
    subgraph_node_mask = torch.tensor([n.item() in nodes.tolist() for n in subgraph_nodes], dtype = torch.bool) \
            if subgraph_nodes is not None else None

    # subgraph_edge_mask = torch.tensor()
    
    return subgraph_node_mask, None
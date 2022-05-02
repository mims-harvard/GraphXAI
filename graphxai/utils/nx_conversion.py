import torch
import numpy as np
import networkx as nx

import torch_geometric.utils as pyg_utils

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
        #data.edge_index = pyg_utils.to_undirected(data.edge_index)
    else:
        G = nx.DiGraph()

    node_list = sorted(torch.unique(data.edge_index).tolist())
    #node_list = np.arange(data.x.shape[0])
    map_norm = {node_list[i]:i for i in range(len(node_list))}
    rev_map_norm = {v:k for k, v in map_norm.items()}
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
        G = nx.relabel_nodes(G, mapping=rev_map_norm)
        return G

def mask_graph(edge_index: torch.Tensor, 
        node_mask: torch.Tensor = None, 
        edge_mask: torch.Tensor = None):
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

def khop_subgraph_nx(
        node_idx: int,
        num_hops: int, 
        G: nx.Graph
    ):
    '''
    Finds k-hop neighborhood in a networkx graph. Uses a BFS of depth num_hops
        on the networkx Graph provided to find edges.

    ..note:: Includes node_idx within subgraph

    Args:
        node_idx (int): Node for which we are to find a subgraph around.
        num_hops (int): Number of hops for which to search.
        G (nx.Graph): Graph on which to find k-hop subgraph

    :rtype: list
        nodes (list): Nodes in the k-hop subgraph
    '''
    edges = list(nx.bfs_edges(G, node_idx, depth_limit = num_hops))
    return list(np.unique(edges))

def match_torch_to_nx_edges(G: nx.Graph, edge_index: torch.Tensor):
    '''
    Gives dictionary matching index in edge_index to G.edges
        - Supports matching for undirected edges
        - Mainly for plotting
    '''

    edges_list = list(G.edges)

    edges_map = dict()

    for i in range(len(edges_list)):
        e1, e2 = edges_list[i]

        # Check e1 -> 0, e2 -> 1
        # cond1 = ((e1 == edge_index[0,:]) & (e2 == edge_index[1,:])).nonzero(as_tuple=True)[0]
        # cond2 = ((e2 == edge_index[0,:]) & (e1 == edge_index[1,:])).nonzero(as_tuple=True)[0]
        cond1 = ((e1 == edge_index[0,:]) & (e2 == edge_index[1,:])).nonzero(as_tuple=True)[0]
        cond2 = ((e2 == edge_index[0,:]) & (e1 == edge_index[1,:])).nonzero(as_tuple=True)[0]
        #print(cond1)

        if cond1.shape[0] > 0:
            edges_map[(e1, e2)] = cond1[0].item()
            edges_map[(e2, e1)] = cond1[0].item()
        elif cond2.shape[0] > 0:
            edges_map[(e1, e2)] = cond2[0].item()
            edges_map[(e2, e1)] = cond2[0].item()
        else:
            raise ValueError('Edge not in graph')

        # if cond1.shape[0] > 0 and cond2.shape[0] > 0:
        #     # Choose smallest
        #     edges_map[(e1, e2)] = min(cond1[0].item(), cond2[0].item())

        # # Check e1 -> 1, e2 -> 0 if the first condition didn't work
        # else:
        #     if cond1.shape[0] == 0:
        #         if cond2.shape[0] > 0:
        #             edges_map[(e2, e1)] = i
        #         else:
        #             #print(e1, e2)
        #             raise ValueError('Edge not in graph')
        #     else:
        #         edges_map[(e1, e2)] = i # Get first instance, don't care about duplicates

    return edges_map

def remove_duplicate_edges(edge_index):
    # Removes duplicate edges from edge_index, making it arbitrarily directed (random positioning):

    new_edge_index = []
    added_nodes = set()
    dict_tracker = dict()

    edge_mask = torch.zeros(edge_index.shape[1], dtype=bool)

    for i in range(edge_index.shape[1]):
        e1 = edge_index[0,i].item()
        e2 = edge_index[1,i].item()
        if e1 in added_nodes:
            if (e2 in dict_tracker[e1]):
                continue
            dict_tracker[e1].append(e2)
        else:
            dict_tracker[e1] = [e2]
            added_nodes.add(e1)
        if e2 in added_nodes:
            if (e1 in dict_tracker[e2]):
                continue
            dict_tracker[e2].append(e1)
        else:
            dict_tracker[e2] = [e1]
            added_nodes.add(e2)

        new_edge_index.append((e1, e2)) # Append only one version
        edge_mask[i] = True
        # Both versions to dict checker
        # if e1 in added_nodes:
            
        # else:
            

        # if e2 in added_nodes:
        #     dict_tracker[e2].append(e1)
        # else:
        #     dict_tracker[e2] = [e1]
        #     added_nodes.add(e2)

    return torch.tensor(new_edge_index).t().contiguous(), edge_mask
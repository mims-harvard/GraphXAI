import torch
import networkx as nx

from graphxai.utils.nx_conversion import khop_subgraph_nx

def motif_id_label(G, num_hops):
    '''
    Gets labels based on motif label in the neighborhood
    '''
    def get_label(node_idx):
        nodes_in_khop = khop_subgraph_nx(node_idx, num_hops, G)
        # For now, sum motif id's in k-hop (min is 0 for no motifs)
        motif_in_khop = torch.sum(torch.unique([G.nodes[ni]['motif_id'] for ni in nodes_in_khop])).item()
        return torch.tensor(motif_in_khop, dtype=torch.long)

    return get_label

def binary_feature_label(G, method = 'median'):
    '''
    Labeling based solely on features, no edge information
        - Keywords can be given based on type of labeling split

    Args:
        G (nx.Graph): Graph on which the nodes are labeled on
        method (str): Method by which to split the features
    '''
    max_node = len(list(G.nodes))
    node_attr = nx.get_node_attributes(G, 'x')
    if method == 'median':
        x1 = [node_attr[i][1] for i in range(max_node)]
        split = torch.median(x1).item()
    def get_label(node_idx):
        return torch.tensor(int(x1[node_idx] > split), dtype=torch.long)

    return get_label

def number_motif_equal_label(G, num_hops, equal_number=1):
    def get_label(node_idx):
        nodes_in_khop = khop_subgraph_nx(node_idx, num_hops, G)
        num_unique_houses = torch.unique([G.nodes[ni]['shape'] \
            for ni in nodes_in_khop if G.nodes[ni]['shape'] > 0 ]).shape[0]
        return torch.tensor(int(num_unique_houses == equal_number), dtype=torch.long)

    return get_label

def bound_graph_label(G: nx.Graph):
    '''
    Args:
        G (nx.Graph): Graph on which the labels are based on
    '''
    sh = nx.get_node_attributes(G, 'shapes_in_khop')
    def get_label(node_idx):
        return torch.tensor(sh[node_idx] - 1, dtype=torch.long)

    return get_label

def logical_edge_feature_label(G, num_hops = None, feature_method = 'median'):

    if feature_method == 'median':
        # Calculate median (as for feature):
        node_attr = nx.get_node_attributes(G, 'x')
        x1 = [node_attr[i][1] for i in range(G.number_of_nodes())]
        split = torch.median(x1).item()

    def get_label(node_idx):
        nodes_in_khop = khop_subgraph_nx(node_idx, num_hops, G)
        num_unique_houses = torch.unique([G.nodes[ni]['shape'] \
            for ni in nodes_in_khop if G.nodes[ni]['shape'] > 0 ]).shape[0]
        return torch.tensor(int(num_unique_houses == 1 and x1[node_idx] > split), dtype=torch.long)

    return get_label
import torch

from typing import Callable


def graph_build_zero_filling(x: torch.Tensor, edge_index: torch.Tensor,
                             node_mask: torch.Tensor):
    """
    Subgraph building through masking the unselected nodes with zero features.
    """
    ret_x = x * node_mask.unsqueeze(1)
    return ret_x, edge_index


def graph_build_split(x: torch.Tensor, edge_index: torch.Tensor,
                      node_mask: torch.Tensor):
    """
    Subgraph building through spliting the selected nodes from the original graph.
    """
    U, V = edge_index
    edge_mask = (node_mask[U] == 1) & (node_mask[V] == 1)
    ret_edge_index = edge_index[:, edge_mask]
    return x, ret_edge_index


def get_graph_build_func(build_method):
    if build_method.lower() == 'zero_filling':
        return graph_build_zero_filling
    elif build_method.lower() == 'split':
        return graph_build_split
    else:
        raise NotImplementedError()


def gnn_score(coalition: list, x: torch.Tensor, edge_index: torch.Tensor,
              value_func: Callable, subgraph_building_method='zero_filling'):
    """
    Get the value of the subgraph with selected nodes.
    """
    mask = torch.zeros(x.shape[0])
    mask[coalition] = 1.0
    subgraph_build_func = get_graph_build_func(subgraph_building_method)
    sub_x, sub_edge_index = subgraph_build_func(x, edge_index, mask)
    score = value_func(sub_x, sub_edge_index)
    return score.item()


def get_selected_nodes(edge_index: torch.Tensor,
                       edge_mask: torch.Tensor,
                       top_k: int):
    """
    Get the nodes of the top k-edge subgraph.

    Args:
        edge_index (torch.Tensor, [2 x m]): edge index of the graph
        edge_mask (torch.Tensor, [m]): edge mask of the graph
        top_k (int): number of edges to include in the subgraph

    Returns:
        selected_nodes: list of the indices of the selected nodes
    """
    sorted_edge_weights = edge_mask.reshape(-1).sort(descending=True)
    threshold = float(sorted_edge_weights.values[min(top_k, edge_mask.shape[0]-1)])
    hard_mask = edge_mask > threshold
    edge_idx_list = torch.where(hard_mask == 1)[0]
    selected_nodes = []
    for edge_idx in edge_idx_list:
        selected_nodes += [edge_index[0][edge_idx].item(), edge_index[1][edge_idx].item()]
    selected_nodes = list(set(selected_nodes))
    return selected_nodes

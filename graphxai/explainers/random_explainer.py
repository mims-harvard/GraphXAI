import torch
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.utils.num_nodes import maybe_num_nodes

from .root_explainer import RootExplainer


class RandomExplainer(RootExplainer):
    """
    Random Explanation for GNNs
    """
    def __init__(self, model):
        """
        Args:
            model (torch.nn.Module): model on which to make predictions
        """
        super().__init__(model)

    def get_explanation_node(self, x: torch.Tensor, node_idx: int,
                             edge_index: torch.Tensor, *_):
        """
        Get the explanation for a node.

        Args:
            x (torch.Tensor, [n x d]): tensor of node features
            node_idx (int): index of the interested node
            edge_index (torch.Tensor, [2 x m]): edge index of the graph

        Return:
            exp (dict):
                exp['feature'] (torch.Tensor, [d]): feature mask explanation
                exp['edge'] (torch.Tensor, [m]): k-hop edge mask explanation
            khop_info (tuple):
                (0) the nodes involved in the subgraph
                (1) the filtered edge_index (torch.Tensor, [2 x m])
                (2) the mapping from node indices in node_idx to their new location
                    (not useful since node_idx contains only one node here)
                (3) the edge mask indicating which edges were preserved (torch.Tensor, [m])
        """
        exp = {'feature': None, 'edge': None}
        exp['feature'] = torch.randn(x[0, :].shape)
        exp['edge'] = torch.randn(edge_index[0, :].shape)

        num_hops = 2
        khop_info = k_hop_subgraph(node_idx, num_hops, edge_index)

        return exp, khop_info

    def get_explanation_graph(self, x: torch.Tensor, edge_index: torch.Tensor,
                              num_nodes : int = None, *_):
        """
        Get the explanation for the whole graph.

        Args:
            x (torch.Tensor, [n x d]): tensor of node features from the entire graph
            edge_index (torch.Tensor, [2 x m]): edge index of entire graph
            num_nodes (int, optional): number of nodes in graph

        Return:
            exp (dict):
                exp['feature'] (torch.Tensor, [n x d]): feature mask explanation
                exp['edge'] (torch.Tensor, [m]): edge mask explanation
            khop_info (tuple):
                (0) the nodes involved in the subgraph
                (1) the filtered edge_index (torch.Tensor, [2 x m])
                (2) the mapping from node indices in node_idx to their new location
                    (not useful since node_idx contains only one node here)
                (3) the edge mask indicating which edges were preserved (torch.Tensor, [m])
        """
        exp = {'feature': None, 'edge': None}

        n = maybe_num_nodes(edge_index, None) if num_nodes is None else num_nodes
        rand_mask = torch.bernoulli(0.5 * torch.ones(n, 1))
        exp['feature'] = rand_mask * torch.randn_like(x)

        exp['edge'] = torch.randn(edge_index[0, :].shape)

        return exp

    def get_explanation_link(self):
        """
        Get the explanation for a link.
        """
        raise NotImplementedError()

from typing import Optional

import torch
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data
from torch_geometric.utils.num_nodes import maybe_num_nodes

from graphxai.explainers._base import _BaseExplainer
from graphxai.utils.constants import EXP_TYPES
from graphxai.utils import Explanation


class RandomExplainer(_BaseExplainer):
    """
    Random Explanation for GNNs
    """
    def __init__(self, model):
        super().__init__(model)

    def get_explanation_node(self, 
            node_idx: int, 
            x: torch.Tensor,
            edge_index: torch.Tensor, 
            num_hops: Optional[int] = None,
            y = None,
            aggregate_node_imp = torch.sum
        ):
        """
        Get the explanation for a node.

        Args:
            node_idx (int): index of the node to be explained
            x (torch.Tensor, [n x d]): tensor of node features
            edge_index (torch.Tensor, [2 x m]): edge index of the graph
            aggregate_node_imp (function, optional): torch function that aggregates
                all node importance feature-wise scores across the enclosing 
                subgraph. Must support `dim` argument. 
                (:default: :obj:`torch.sum`)

        Returns:
            exp (dict):
                exp['feature_imp'] (torch.Tensor, [d]): feature mask explanation
                exp['edge_imp'] (torch.Tensor, [m]): k-hop edge importance
                exp['node_imp'] (torch.Tensor, [m]): k-hop node importance
            khop_info (4-tuple of torch.Tensor):
                0. the nodes involved in the subgraph
                1. the filtered `edge_index`
                2. the mapping from node indices in `node_idx` to their new location
                3. the `edge_index` mask indicating which edges were preserved
        """
        num_hops = self.L if num_hops is None else num_hops
        khop_info = k_hop_subgraph(node_idx, num_hops, edge_index)

        # exp = {k: None for k in EXP_TYPES}
        # exp['feature_imp'] = torch.randn(x[0, :].shape)
        # exp['edge_imp'] = torch.randn(edge_index[0, :].shape)

        #node_imp = aggregate_node_imp(torch.randn(khop_info[0].shape), dim = 1)
        
        exp = Explanation(
            feature_imp = torch.randn(x[0, :].shape),
            node_imp = torch.randn(khop_info[0].shape),
            node_idx = node_idx
        )

        exp.set_enclosing_subgraph(khop_info)

        return exp

    def get_explanation_graph(self, 
            x: torch.Tensor, 
            edge_index: torch.Tensor,
            num_nodes : int = None,
            aggregate_node_imp = torch.sum):
        """
        Get the explanation for the whole graph.

        Args:
            x (torch.Tensor, [n x d]): tensor of node features from the entire graph
            edge_index (torch.Tensor, [2 x m]): edge index of entire graph
            num_nodes (int, optional): number of nodes in graph
            aggregate_node_imp (function, optional): torch function that aggregates
                all node importance feature-wise scores across the graph. 
                Must support `dim` argument. (:default: :obj:`torch.sum`)

        Returns:
            exp (dict):
                exp['feature_imp'] (torch.Tensor, [d]): feature mask explanation
                exp['edge_imp'] (torch.Tensor, [m]): k-hop edge importance
                exp['node_imp'] (torch.Tensor, [m]): k-hop node importance
        """
        #exp = {k: None for k in EXP_TYPES}

        n = maybe_num_nodes(edge_index, None) if num_nodes is None else num_nodes
        rand_mask = torch.bernoulli(0.5 * torch.ones(n, 1))
        # exp['feature_imp'] = rand_mask * torch.randn_like(x)

        # exp['edge_imp'] = torch.randn(edge_index[0, :].shape)

        node_imp = aggregate_node_imp(rand_mask * torch.randn_like(x), dim=1)

        exp = Explanation(
            node_imp = node_imp,
            edge_imp = torch.randn(edge_index[0, :].shape)
        )

        exp.set_whole_graph(Data(x=x, edge_index = edge_index))

        return exp

    def get_explanation_link(self):
        """
        Explain a link prediction.
        """
        raise NotImplementedError()

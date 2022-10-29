from typing import Optional, Callable
#from collections.abc import Callable

import torch
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data
from torch_geometric.utils.num_nodes import maybe_num_nodes

from graphxai.explainers._base import _BaseExplainer
from graphxai.utils import Explanation

class RandomExplainer(_BaseExplainer):
    """
    Random Explanation for GNNs

    Args:
        model (torch.nn.Module): Model for which to explain.
    """
    def __init__(self, model: torch.nn.Module):
        super().__init__(model)

    def get_explanation_node(self, 
            node_idx: int, 
            x: torch.Tensor,
            edge_index: torch.Tensor, 
            num_hops: Optional[int] = None,
            node_agg: Optional[Callable[[torch.Tensor], torch.Tensor]] = torch.sum,
            forward_kwargs: dict = {}
        ):
        """
        Get the explanation for a node.

        Args:
            node_idx (int): index of the node to be explained
            x (torch.Tensor, [n x d]): tensor of node features
            edge_index (torch.Tensor, [2 x m]): edge index of the graph
            num_hops (int, optional): Number of hops for the enclosing subgraph.
                None means that this value is computed automatically from the model.
                (:default: :obj:`None`) 
            node_agg (function, optional): torch function that aggregates
                all node importance feature-wise scores across the enclosing 
                subgraph. Must support `dim` argument. 
                (:default: :obj:`torch.sum`)
            forward_kwargs (dict, optional): Has no effect; provided for consistency
                with other methods. (:default: :obj:`None`)

        Returns:
            exp (:class:`Explanation`): Explanation output from the method.
                Fields are:
                `feature_imp`: :obj:`torch.Tensor, [x.shape[1],]`
                `node_imp`: :obj:`torch.Tensor, [nodes_in_khop,]`
                `edge_imp`: :obj:`torch.Tensor, [edges_in_khop,]`
                `enc_subgraph`: :obj:`graphxai.utils.EnclosingSubgraph`
        """
        num_hops = self.L if num_hops is None else num_hops
        khop_info = k_hop_subgraph(node_idx, num_hops, edge_index)
        
        # Generate node mask and random values on each node
        n = khop_info[0].shape[0]
        rand_mask = torch.bernoulli(0.5 * torch.ones(n, 1)).to(x.device)
        randn = torch.randn(n).to(x.device)
        node_imp = node_agg(rand_mask * randn, dim=1)
        
        exp = Explanation(
            feature_imp = torch.randn(x[0, :].shape),
            node_imp = node_imp,
            edge_imp = torch.randn(khop_info[1][0, :].shape), # Random mask over edges
            node_idx = node_idx
        )

        exp.set_enclosing_subgraph(khop_info)

        return exp

    def get_explanation_graph(self, 
            x: torch.Tensor, 
            edge_index: torch.Tensor,
            num_nodes : int = None,
            node_agg: Optional[Callable[[torch.Tensor], torch.Tensor]] = torch.sum,
            forward_kwargs: dict = {}):
        """
        Get the explanation for the whole graph.

        Args:
            x (torch.Tensor, [n x d]): tensor of node features from the entire graph
            edge_index (torch.Tensor, [2 x m]): edge index of entire graph
            num_nodes (int, optional): number of nodes in graph
            node_agg (function, optional): torch function that aggregates
                all node importance feature-wise scores across the graph. 
                Must support `dim` argument. (:default: :obj:`torch.sum`)
            forward_kwargs (dict, optional): Has no effect; provided for consistency
                with other methods. (:default: :obj:`None`)

        Returns:
            exp (:class:`Explanation`): Explanation output from the method.
                Fields are:
                `feature_imp`: :obj:`torch.Tensor, [x.shape[1],]`
                `node_imp`: :obj:`torch.Tensor, [nodes,]`
                `edge_imp`: :obj:`torch.Tensor, [edge_index.shape[1],]`
                `graph`: :obj:`torch_geometric.data.Data`
        """
        n = maybe_num_nodes(edge_index, None) if num_nodes is None else num_nodes
        rand_mask = torch.bernoulli(0.5 * torch.ones(n, 1)).to(x.device)

        randn = torch.randn_like(x).to(x.device)

        node_imp = node_agg(rand_mask * randn, dim=1)

        exp = Explanation(
            feature_imp = torch.randn(x.shape[0]),
            node_imp = node_imp,
            edge_imp = torch.randn(edge_index[0, :].shape).to(edge_index.device)
        )

        exp.set_whole_graph(Data(x=x, edge_index = edge_index))

        return exp

    def get_explanation_link(self):
        """
        Explain a link prediction.
        """
        raise NotImplementedError()

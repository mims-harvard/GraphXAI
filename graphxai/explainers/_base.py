import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import k_hop_subgraph


class _BaseExplainer:
    """
    Base Class for Explainers
    """
    def __init__(self, model: nn.Module, criterion: nn.Module, *_):
        """
        Args:
            model (torch.nn.Module): model on which to make predictions
                The output of the model should be unnormalized class score.
                For example, last layer = GCNConv or Linear.
            criterion (torch.nn.Module): loss function
        """
        self.model = model
        self.criterion = criterion
        self.L = len([module for module in self.model.modules()
                      if isinstance(module, MessagePassing)])

    def _predict(self, x: torch.Tensor,
                 edge_index: torch.Tensor,
                 return_type: str = 'label',
                 forward_kwargs: dict = {}):
        """
        Get the model's prediction.

        Args:
            edge_index (torch.Tensor, [2 x m]): edge index of the graph
            x (torch.Tensor, [n x d]): node features
            return_type (str): one of ['label', 'logit', 'log_logit']
            forward_kwargs (dict, optional): additional arguments to model.forward
                beyond x and edge_index
        """
        # Compute unnormalized class score
        with torch.no_grad():
            output = self.model(x, edge_index, **forward_kwargs)

        if return_type == 'label':
            return output.argmax(axis=1)
        elif return_type == 'logit':
            return F.softmax(output)
        elif return_type == 'log_logit':
            return F.log_softmax(output)
        else:
            raise ValueError("return_type must be 'label', 'logit', or 'log_logit'")

    def _subgraph(self):
        """
        # TODO: Design this function
        # Return numpy version
        """
        khop_info = subset, sub_edge_index, mapping, _ = \
            k_hop_subgraph(node_idx, num_hops, edge_index,
                           relabel_nodes=True, num_nodes=x.shape[0])
        return khop_info

    def get_explanation_node(self, node_idx: int,
                             x: torch.Tensor,
                             edge_index: torch.Tensor,
                             label: Optional[torch.Tensor] = None,
                             num_hops: Optional[int] = None,
                             forward_kwargs: dict = {}, *_):
        """
        Explain a node prediction.

        Args:
            node_idx (int): index of the node to be explained
            x (torch.Tensor, [n x d]): node features
            edge_index (torch.Tensor, [2 x m]): edge index of the graph
            label (torch.Tensor, optional, [n x ...]): labels to explain
                If not provided, we use the output of the model.
            num_hops (int, optional): number of hops to consider
                If not provided, we use the number of graph layers of the GNN.
            forward_kwargs (dict, optional): additional arguments to model.forward
                beyond x and edge_index

        Returns:
            exp (dict):
                exp['feature'] (torch.Tensor, [d]): feature mask explanation
                exp['edge'] (torch.Tensor, [m]): k-hop edge mask explanation
            khop_info (4-tuple of torch.Tensor):
                0. the nodes involved in the subgraph
                1. the filtered `edge_index`
                2. the mapping from node indices in `node_idx` to their new location
                3. the `edge_index` mask indicating which edges were preserved
        """
        # If labels are needed
        label = self._predict(x, edge_index, return_type='label') if label is None else label
        # If logits / log logits are needed
        logit = self._predict(x, edge_index, return_type='logit')
        log_logit = self._predict(x, edge_index, return_type='log_logit')

        num_hops = self.L if num_hops is None else num_hops

        khop_info = subset, sub_edge_index, mapping, _ = \
            k_hop_subgraph(node_idx, num_hops, edge_index,
                           relabel_nodes=True, num_nodes=x.shape[0])
        sub_x = x[subset]

        exp = {'feature': None, 'edge': None}

        # Compute exp
        raise NotImplementedError()

        return exp, khop_info

    def get_explanation_graph(self, edge_index: torch.Tensor,
                              x: torch.Tensor, label: torch.Tensor,
                              forward_kwargs: Optional[dict] = None, *_):
        """
        Explain a whole-graph prediction.

        Args:
            edge_index (torch.Tensor, [2 x m]): edge index of the graph
            x (torch.Tensor, [n x d]): node features
            label (torch.Tensor, [n x ...]): labels to explain
            forward_kwargs (dict, optional): additional arguments to model.forward
                beyond x and edge_index

        Returns:
            exp (dict):
                exp['feature'] (torch.Tensor, [n x d]): feature mask explanation
                exp['edge'] (torch.Tensor, [m]): k-hop edge mask explanation
        """
        exp = {'feature': None, 'edge': None}

        # Compute exp
        raise NotImplementedError()

        return exp

    def get_explanation_link(self):
        """
        Explain a link prediction.
        """
        raise NotImplementedError()

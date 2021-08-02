import numpy as np
import torch

from typing import Optional
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import k_hop_subgraph

from ._base import _BaseExplainer


class GNNExplainer(_BaseExplainer):
    """
    GNNExplainer: node only
    """
    def __init__(self, model, coeff=None):
        """
        Args:
            model (torch.nn.Module): model on which to make predictions
                The output of the model should be unnormalized class score.
                For example, last layer = CNConv or Linear.
            coeff (dict, optional): coefficient of the entropy term and the size term
                for learning edge mask and node feature mask
                Default setting:
                    coeff = {'edge': {'entropy': 1.0, 'size': 0.005},
                             'feature': {'entropy': 0.1, 'size': 1.0}}
        """
        super().__init__(model)
        if coeff is not None:
            self.coeff = coeff
        else:
            self.coeff = {'edge': {'entropy': 1.0, 'size': 0.005},
                          'feature': {'entropy': 0.1, 'size': 1.0}}

    def get_explanation_node(self, node_idx: int, edge_index: torch.Tensor,
                             x: torch.Tensor, label: Optional[torch.Tensor] = None,
                             num_hops: Optional[int] = None):
        """
        Explain a node prediction.

        Args:
            node_idx (int): index of the node to be explained
            edge_index (torch.Tensor, [2 x m]): edge index of the graph
            x (torch.Tensor, [n x d]): node features
            label (torch.Tensor, optional, [n x ...]): labels to explain
                If not provided, we use the output of the model.
            num_hops (int, optional): number of hops to consider
                If not provided, we use the number of graph layers of the GNN.

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
        label = self._predict(x, edge_index) if label is None else label
        num_hops = self.L if num_hops is None else num_hops

        exp = {'feature': None, 'edge': None}

        khop_info = subset, sub_edge_index, mapping, _ = \
            k_hop_subgraph(node_idx, num_hops, edge_index,
                           relabel_nodes=True, num_nodes=x.shape[0])
        sub_x = x[subset]
        num_features = x.shape[1]
        sub_num_nodes = subset.shape[0]
        sub_num_edges = sub_edge_index.shape[1]

        # Initialize edge_mask and feature_mask for learning
        std = torch.nn.init.calculate_gain('relu') * np.sqrt(2.0 / (2 * sub_num_nodes))
        edge_mask = torch.nn.Parameter(torch.randn(sub_num_edges) * std)
        feature_mask = torch.nn.Parameter(torch.randn(num_features) * 0.1)

        self.model.eval()
        num_epochs = 200
        optimizer = torch.optim.Adam([edge_mask], lr=0.01)

        # Loss function for GNNExplainer's objective
        def loss_fn(logit, mask, mask_type):
            # Select the logit and the label of node_idx
            node_logit = logit[torch.where(subset==node_idx)].squeeze()
            node_label = label[mapping]
            # Select the label's logit value
            loss = node_logit[node_label].item()
            # Q: 1) Why this logit term? 2) No joint learning of feature and edge masks
            a = mask.sigmoid()
            loss = loss + self.coeff[mask_type]['size'] * torch.sum(a)
            entropy = -a * torch.log(a + 1e-15) - (1-a) * torch.log(1-a + 1e-15)
            loss = loss + self.coeff[mask_type]['entropy'] * entropy.mean()
            return loss

        def train(mask, mask_type):
            optimizer = torch.optim.Adam([mask], lr=0.01)
            for epoch in range(1, num_epochs+1):
                optimizer.zero_grad()
                logit = self._predict(sub_x, sub_edge_index, return_type='logit')
                loss = loss_fn(logit, mask, mask_type)
                loss.backward()
                optimizer.step()

        train(feature_mask, 'feature')
        train(edge_mask, 'edge')

        exp['feature'] = feature_mask.data
        exp['edge'] = edge_mask.data

        return exp, khop_info

    def get_explanation_graph(self, edge_index: torch.Tensor,
                              x: torch.Tensor, label: torch.Tensor,
                              forward_kwargs=None):
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
        raise Exception('GNNExplainer cannot provide explanation \
                         for graph-level prediction.')

    def get_explanation_link(self):
        """
        Explain a link prediction.
        """
        raise NotImplementedError()

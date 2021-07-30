import torch

from typing import Optional
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import k_hop_subgraph


class GradExplainer:
    """
    Vanilla Gradient Explanation for GNNs
    """
    def __init__(self, model, criterion, *_):
        """
        Args:
            model (torch.nn.Module): model on which to make predictions
                The output of the model should be unnormalized class score.
                For example, last layer = CNConv or Linear.
            criterion (torch.nn.Module): loss function
        """
        self.model = model
        self.criterion = criterion
        self.L = len([module for module in self.model.modules()
                      if isinstance(module, MessagePassing)])

    def get_explanation_node(self, node_idx: int, edge_index: torch.Tensor,
                             x: torch.Tensor, label: Optional[torch.Tensor] = None,
                             num_hops: Optional[int] = None, *_):
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
        label = self._predict(x) if label is None else label
        num_hops = self.L if num_hops is None else num_hops

        exp = {'feature': None, 'edge': None}

        khop_info = subset, sub_edge_index, mapping, _ = \
            k_hop_subgraph(node_idx, num_hops, edge_index,
                           relabel_nodes=True, num_nodes=x.shape[0])
        sub_x = x[subset]

        self.model.eval()
        sub_x.requires_grad = True
        output = self.model(sub_x, sub_edge_index)
        loss = self.criterion(output[mapping], label[mapping])
        loss.backward()

        exp['feature'] = sub_x.grad[torch.where(subset == node_idx)[0].item(), :]

        return exp, khop_info

    def get_explanation_graph(self, edge_index: torch.Tensor,
                              x: torch.Tensor, label: torch.Tensor,
                              forward_kwargs=None, *_):
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

        self.model.eval()
        x.requires_grad = True
        if forward_kwargs is None:
            output = self.model(x, edge_index)
        else:
            output = self.model(x, edge_index, **forward_kwargs)
        loss = self.criterion(output, label)
        loss.backward()

        exp['feature'] = x.grad

        return exp

    def get_explanation_link(self):
        """
        Explain a link prediction.
        """
        raise NotImplementedError()

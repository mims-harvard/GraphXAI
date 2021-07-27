import torch
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.utils.num_nodes import maybe_num_nodes

from .root_explainer import RootExplainer


class GradExplainer(RootExplainer):
    """
    Vanilla Gradient Explanation for GNNs
    """
    def __init__(self, model, criterion):
        """
        Args:
            model (torch.nn.Module): model on which to make predictions
            criterion (torch.nn.Module): loss function
        """
        super().__init__(model, criterion)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def get_explanation_node(self, node_idx: int, edge_index: torch.Tensor,
                             x: torch.Tensor, label: torch.Tensor, *_):
        """
        Explain a node prediction.

        Args:
            node_idx (int): index of the node to be explained
            edge_index (torch.Tensor, [2 x m]): edge index of the graph
            x (torch.Tensor, [n x d]): node features
            label (torch.Tensor, [n x ...]): labels to explain

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
        exp = {'feature': None, 'edge': None}

        num_hops = self.L
        khop_info = subset, sub_edge_index, mapping, _ = \
            k_hop_subgraph(node_idx, num_hops, edge_index,
                           relabel_nodes=True, num_nodes=x.shape[0])
        sub_x = x[subset]

        self.model.eval()
        sub_x.requires_grad = True
        output = self.model(sub_x.to(self.device), sub_edge_index.to(self.device))
        loss = self.criterion(output[mapping], label[mapping].to(self.device))
        loss.backward()

        exp['feature'] = sub_x.grad[torch.where(subset == node_idx)[0].item(), :]

        return exp, khop_info

    def get_explanation_graph(self, edge_index: torch.Tensor,
                              x: torch.Tensor, label: torch.Tensor,
                              forward_args=None, *_):
        """
        Explain a whole-graph prediction.

        Args:
            edge_index (torch.Tensor, [2 x m]): edge index of the graph
            x (torch.Tensor, [n x d]): node features
            label (torch.Tensor, [n x ...]): labels to explain
            forward_args (tuple, optional): additional arguments to model.forward
                beyond x and edge_index

        Returns:
            exp (dict):
                exp['feature'] (torch.Tensor, [n x d]): feature mask explanation
                exp['edge'] (torch.Tensor, [m]): k-hop edge mask explanation
            khop_info (4-tuple of torch.Tensor):
                0. the nodes involved in the subgraph
                1. the filtered `edge_index`
                2. the mapping from node indices in `node_idx` to their new location
                3. the `edge_index` mask indicating which edges were preserved
        """
        exp = {'feature': None, 'edge': None}

        self.model.eval()
        x.requires_grad = True
        if forward_args is None:
            output = self.model(x.to(self.device), edge_index.to(self.device))
        else:
            output = self.model(x.to(self.device), edge_index.to(self.device), *forward_args)
        loss = self.criterion(output, label.to(self.device))
        loss.backward()

        exp['feature'] = x.grad

        return exp

    def get_explanation_link(self):
        """
        Get the explanation for a link.
        """
        raise NotImplementedError()

import torch

from typing import Optional, Callable
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data

from graphxai.explainers._base import _BaseExplainer
from graphxai.utils import Explanation


class GradExplainer(_BaseExplainer):
    """
    Vanilla Gradient Explanation for GNNs

    Args:
        model (torch.nn.Module): model on which to make predictions
            The output of the model should be unnormalized class score.
            For example, last layer = CNConv or Linear.
        criterion (torch.nn.Module): loss function
    """
    def __init__(self, model: torch.nn.Module, 
            criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        super().__init__(model)
        self.criterion = criterion

    def get_explanation_node(self, node_idx: int, x: torch.Tensor,
                             edge_index: torch.Tensor,
                             label: Optional[torch.Tensor] = None,
                             num_hops: Optional[int] = None,
                             aggregate_node_imp = torch.sum,
                             y = None,
                             forward_kwargs: dict = {}, **_) -> Explanation:
        """
        Explain a node prediction.

        Args:
            node_idx (int): index of the node to be explained
            x (torch.Tensor, [n x d]): node features
            edge_index (torch.Tensor, [2 x m]): edge index of the graph
            label (torch.Tensor, optional, [n x ...]): labels to explain
                If not provided, we use the output of the model.
                (:default: :obj:`None`)
            num_hops (int, optional): number of hops to consider
                If not provided, we use the number of graph layers of the GNN.
                (:default: :obj:`None`)
            aggregate_node_imp (function, optional): torch function that aggregates
                all node importance feature-wise scores across the enclosing 
                subgraph. Must support `dim` argument. 
                (:default: :obj:`torch.sum`)
            forward_kwargs (dict, optional): Additional arguments to model.forward 
                beyond x and edge_index. Must be keyed on argument name. 
                (default: :obj:`{}`)

        :rtype: :class:`graphxai.Explanation`

        Returns:
            exp (:class:`Explanation`): Explanation output from the method.
                Fields are:
                `feature_imp`: :obj:`torch.Tensor, [features,]`
                `node_imp`: :obj:`torch.Tensor, [nodes_in_khop, features]`
                `edge_imp`: :obj:`None`
                `enc_subgraph`: :obj:`graphxai.utils.EnclosingSubgraph`
        """
        label = self._predict(x, edge_index,
                              forward_kwargs=forward_kwargs) if label is None else label
        num_hops = self.L if num_hops is None else num_hops

        khop_info = subset, sub_edge_index, mapping, _ = \
            k_hop_subgraph(node_idx, num_hops, edge_index,
                           relabel_nodes=True, num_nodes=x.shape[0])
        sub_x = x[subset]

        self.model.eval()
        sub_x.requires_grad = True
        output = self.model(sub_x, sub_edge_index)
        loss = self.criterion(output[mapping], label[mapping])
        loss.backward()

        feature_imp = sub_x.grad[torch.where(subset == node_idx)].squeeze(0)
        node_imp = aggregate_node_imp(sub_x.grad, dim = 1)

        exp = Explanation(
            feature_imp = feature_imp, #[score_1, ]
            node_imp = node_imp, #[score_1, score_2, ...] [[], []] NxF
            node_idx = node_idx
        )

        exp.set_enclosing_subgraph(khop_info)

        return exp

    def get_explanation_graph(self, 
                                x: torch.Tensor, 
                                edge_index: torch.Tensor,
                                label: torch.Tensor, 
                                aggregate_node_imp = torch.sum,
                                forward_kwargs: dict = {}) -> Explanation:
        """
        Explain a whole-graph prediction.

        Args:
            x (torch.Tensor, [n x d]): node features
            edge_index (torch.Tensor, [2 x m]): edge index of the graph
            label (torch.Tensor, [n x ...]): labels to explain
            aggregate_node_imp (function, optional): torch function that aggregates
                all node importance feature-wise scores across the graph. 
                Must support `dim` argument. (:default: :obj:`torch.sum`)
            forward_kwargs (dict, optional): additional arguments to model.forward
                beyond x and edge_index

        :rtype: :class:`graphxai.Explanation`

        Returns:
            exp (:class:`Explanation`): Explanation output from the method. 
                Fields are:
                `feature_imp`: :obj:`None`
                `node_imp`: :obj:`torch.Tensor, [num_nodes, features]`
                `edge_imp`: :obj:`None`
                `graph`: :obj:`torch_geometric.data.Data`
        """

        self.model.eval()
        x.requires_grad = True
        output = self.model(x, edge_index, **forward_kwargs)
        loss = self.criterion(output, label)
        loss.backward()

        node_imp = aggregate_node_imp(x.grad, dim = 1)

        exp = Explanation(
            node_imp = node_imp
        )

        exp.set_whole_graph(Data(x=x, edge_index=edge_index))

        return exp

    def get_explanation_link(self):
        """
        Explain a link prediction.
        """
        raise NotImplementedError()

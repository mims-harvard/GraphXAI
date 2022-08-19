import torch
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data
from typing import Optional, Callable

from graphxai.explainers._base import _BaseExplainer
from graphxai.utils import Explanation

device = "cuda" if torch.cuda.is_available() else "cpu"


class IntegratedGradExplainer(_BaseExplainer):
    """
    Integrated Gradient Explanation for GNNs

    Args:
        model (torch.nn.Module): Model on which to make predictions.
        criterion (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): Loss function.
    """
    def __init__(self, model: torch.nn.Module, 
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        super().__init__(model)
        self.criterion = criterion

    def get_explanation_node(self, node_idx: int, 
            x: torch.Tensor,
            edge_index: torch.Tensor, 
            label: Optional[torch.Tensor] = None,
            y: Optional[torch.Tensor] = None,
            num_hops: Optional[int] = None, 
            steps: Optional[int] = 40,
            **_):
        """
        Explain a node prediction.

        Args:
            node_idx (int): Index of the node to be explained.
            edge_index (torch.Tensor, [2 x m]): Edge index of the graph.
            x (torch.Tensor, [n x d]): Node features.
            label (torch.Tensor, [n x ...]): Labels to explain.
            y (torch.Tensor): Same as `label`, provided for general 
                compatibility in the arguments. (:default: :obj:`None`)
            num_hops (int, optional): Number of hops in the enclosing 
                subgraph. If `None`, set to the number of layers in 
                the GNN. (:default: :obj:`None`)
            steps (int, optional): Number of steps for the Riemannian 
                integration. (:default: :obj:`40`)

        Returns:
            exp (:class:`Explanation`): Explanation output from the method.
                Fields are:
                `feature_imp`: :obj:`torch.Tensor, [x.shape[1],]`
                `node_imp`: :obj:`torch.Tensor, [nodes_in_khop,]`
                `edge_imp`: :obj:`None`
                `enc_subgraph`: :class:`graphxai.utils.EnclosingSubgraph`
        """

        if (label is None) and (y is None):
            raise ValueError('Either label or y should be provided for Integrated Gradients')

        label = y[node_idx] if label is None else label 
        if len(label.shape) == 0:
            label = label.unsqueeze(dim=0)

        num_hops = num_hops if num_hops is not None else self.L
        khop_info = subset, sub_edge_index, mapping, _ = \
            k_hop_subgraph(node_idx, num_hops, edge_index,
                           relabel_nodes=True, num_nodes=x.shape[0])
        sub_x = x[subset]

        self.model.eval()
        grads = torch.zeros(steps+1, x.shape[1]).to(device)

        # Perform Riemannian integration
        for i in range(steps+1):
            with torch.no_grad():
                baseline = torch.zeros_like(sub_x).to(device)  # TODO: baseline all 0s, all 1s, ...?
                temp_x = baseline + (float(i)/steps) * (sub_x.clone()-baseline)
            temp_x.requires_grad = True
            output = self.model(temp_x, sub_edge_index)
            loss = self.criterion(output[mapping], label)
            loss.backward()
            grad = temp_x.grad[torch.where(subset==node_idx)[0].item()]
            grads[i] = grad

        grads = (grads[:-1] + grads[1:]) / 2.0
        avg_grads = torch.mean(grads, axis=0)

        # Integrated gradients for only node_idx:
        # baseline[0] just gets a single-value 0-tensor
        integrated_gradients = ((x[torch.where(subset == node_idx)[0].item()]
                                 - baseline[0]) * avg_grads)

        # Integrated gradients across the enclosing subgraph:
        all_node_ig = ((x[subset] - baseline) * avg_grads)

        exp = Explanation(
            feature_imp = integrated_gradients,
            node_imp = torch.sum(all_node_ig, dim=1),
            node_idx = node_idx
        )
        exp.set_enclosing_subgraph(khop_info)

        return exp

    def get_explanation_graph(self, edge_index: torch.Tensor,
                              x: torch.Tensor, 
                              label: torch.Tensor = None,
                              y: torch.Tensor = None,
                              node_agg = torch.sum,
                              steps: int = 40,
                              forward_kwargs={}):
        """
        Explain a whole-graph prediction.

        Args:
            edge_index (torch.Tensor, [2 x m]): edge index of the graph
            x (torch.Tensor, [n x d]): node features
            label (torch.Tensor, [n x ...]): labels to explain
            y (torch.Tensor): Same as `label`, provided for general 
                compatibility in the arguments. (:default: :obj:`None`)
            node_agg : 
            forward_args (tuple, optional): additional arguments to model.forward
                beyond x and edge_index

        Returns:
            exp (:class:`Explanation`): Explanation output from the method.
                Fields are:
                `feature_imp`: :obj:`None`
                `node_imp`: :obj:`torch.Tensor, [nodes_in_khop,]`
                `edge_imp`: :obj:`torch.Tensor, [edge_index.shape[1],]`
                `graph`: :obj:`torch_geometric.data.Data`
        """

        if (label is None) and (y is None):
            raise ValueError('Either label or y should be provided for Integrated Gradients')

        label = y if label is None else label 

        self.model.eval()
        grads = torch.zeros(steps+1, *x.shape).to(x.device)
        baseline = torch.zeros_like(x).to(x.device)  # TODO: baseline all 0s, all 1s, ...?
        for i in range(steps+1):
            with torch.no_grad():
                temp_x = baseline + (float(i)/steps) * (x.clone()-baseline)
            temp_x.requires_grad = True
            if forward_kwargs is None:
                output = self.model(temp_x, edge_index)
            else:
                output = self.model(temp_x, edge_index, **forward_kwargs)
            loss = self.criterion(output, label)
            loss.backward()
            grad = temp_x.grad
            grads[i] = grad

        grads = (grads[:-1] + grads[1:]) / 2.0
        avg_grads = torch.mean(grads, axis=0)
        integrated_gradients = (x - baseline) * avg_grads

        exp = Explanation(
            node_imp = node_agg(integrated_gradients, dim=1),
        )

        exp.set_whole_graph(Data(x=x, edge_index=edge_index))

        return exp

    def get_explanation_link(self):
        """
        Explain a link prediction.
        """
        raise NotImplementedError()

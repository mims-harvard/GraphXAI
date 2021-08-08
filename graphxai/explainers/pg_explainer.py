import torch
import torch.nn as nn

from typing import Optional
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import k_hop_subgraph

from graphxai.explainers._base import _BaseExplainer
from graphxai.utils.constants import EXP_TYPES


class PGExplainer(_BaseExplainer):
    """
    PGExplainer

    Code adapted from DIG
    """
    def __init__(self, model: nn.Module, in_channels: int,
                 coeff_size: float = 0.01, coeff_ent: float = 5e-4,
                 t0: float = 5.0, t1: float = 1.0,
                 lr: float = 0.005, epochs: int = 20, init_bias: float = 0.0):
        """
        Args:
            model (torch.nn.Module): model on which to make predictions
                The output of the model should be unnormalized class score.
                For example, last layer = CNConv or Linear.
            in_channels: number of input channels for the explanation network
            coeff_size (float): size regularization to constrain the explanation size
            coeff_ent (float): entropy regularization to constrain the connectivity of explanation
            t0 (float): the temperature at the first epoch
            t1(float): the temperature at the final epoch
            lr (float): learning rate to train the explanation network
            epochs (int): number of epochs to train the explanation network
        """
        super().__init__(model)
        self.in_channels = in_channels

        # Parameters for PGExplainer
        self.coeff_size = coeff_size
        self.coeff_ent = coeff_ent
        self.t0 = t0
        self.t1 = t1
        self.lr = lr
        self.epochs = epochs
        self.init_bias = init_bias

        # Explanation model in PGExplainer
        self.elayers = nn.ModuleList(
            [nn.Sequential(nn.Linear(in_channels, 64), nn.ReLU()),
             nn.Linear(64, 1)])
        self.elayers.append(nn.Sequential(nn.Linear(in_channels, 64), nn.ReLU()))
        self.elayers.append(nn.Linear(64, 1))

    def __concrete_sample(self, log_alpha: torch.Tensor,
                          beta: float = 1.0, training: bool = True):
        """
        Sample from the instantiation of concrete distribution when training.

        Returns:
            training == True: sigmoid(log_alpha + noise)
            training == False: sigmoid(log_alpha)
        """
        if training:
            random_noise = torch.rand(log_alpha.shape)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            h = (random_noise + log_alpha) / beta
            output = h.sigmoid()
        else:
            output = log_alpha.sigmoid()

        return output

    def __emb_to_edge_mask(self, edge_index: torch.Tensor, x: torch.Tensor,
                           emb: torch.Tensor, tmp: float = 1.0, training: bool = False,
                           explain_graph: bool = False, node_idx: int = None):
        """
        Compute the edge mask based on embedding.

        Returns:
            logit (torch.Tensor): the predicted probability when applying edge_mask
            edge_mask (torch.Tensor): the mask for graph edges with values in [0, 1]
        """
        if not explain_graph and node_idx is None:
            raise ValueError('node_idx should be provided.')

        # Concat relevant node embeddings
        U, V = edge_index  # edge (u, v), U = (u), V = (v)
        h1 = emb[U]
        h2 = emb[V]
        if explain_graph:
            h = torch.cat([h1, h2], dim=1)
        else:
            h3 = emb[node_idx].repeat(h1.shape[0], 1)
            h = torch.cat([h1, h2, h3], dim=1)

        # Calculate the edge weights (edge mask)
        for elayer in self.elayers:
            h = elayer(h)
        h = h.squeeze()
        edge_weights = self.__concrete_sample(h, tmp, training)
        num_nodes = emb.shape[0]
        mask_sparse = torch.sparse_coo_tensor(
            edge_index, edge_weights, (num_nodes, num_nodes))
        mask_sigmoid = mask_sparse.to_dense()  # Only applicable to small graphs
        sym_mask = (mask_sigmoid + mask_sigmoid.transpose(0, 1)) / 2  # Undirected only
        edge_mask = sym_mask[edge_index[0], edge_index[1]]

        # Compute the model prediction with edge mask
        logit = self._predict(x, edge_index, return_type='logit')

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
                exp['feature_imp'] (torch.Tensor, [d]): feature mask explanation
                exp['edge_imp'] (torch.Tensor, [m]): k-hop edge importance
                exp['node_imp'] (torch.Tensor, [m]): k-hop node importance
            khop_info (4-tuple of torch.Tensor):
                0. the nodes involved in the subgraph
                1. the filtered `edge_index`
                2. the mapping from node indices in `node_idx` to their new location
                3. the `edge_index` mask indicating which edges were preserved
        """
        label = self._predict(x, edge_index) if label is None else label
        num_hops = self.L if num_hops is None else num_hops

        exp = {k: None for k in EXP_TYPES}

        khop_info = subset, sub_edge_index, mapping, _ = \
            k_hop_subgraph(node_idx, num_hops, edge_index,
                           relabel_nodes=True, num_nodes=x.shape[0])
        sub_x = x[subset]

        self.model.eval()
        sub_x.requires_grad = True
        output = self.model(sub_x, sub_edge_index)
        loss = self.criterion(output[mapping], label[mapping])
        loss.backward()

        exp['feature_imp'] = sub_x.grad[torch.where(subset == node_idx)].squeeze(0)

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
                exp['feature_imp'] (torch.Tensor, [d]): feature mask explanation
                exp['edge_imp'] (torch.Tensor, [m]): k-hop edge importance
                exp['node_imp'] (torch.Tensor, [m]): k-hop node importance

        """
        exp = {k: None for k in EXP_TYPES}

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

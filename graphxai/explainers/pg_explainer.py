import numpy as np
import torch
import torch.nn as nn
import tqdm
import time

from typing import Optional
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data

from graphxai.explainers._base import _BaseExplainer
from graphxai.utils.subgraph import get_selected_nodes, gnn_score
from graphxai.utils.constants import EXP_TYPES


class PGExplainer(_BaseExplainer):
    """
    PGExplainer

    Code adapted from DIG
    """
    def __init__(self, model: nn.Module, emb_layer_name: str = None,
                 explain_graph: bool = False, num_hops: int = 3,
                 coeff_size: float = 0.01, coeff_ent: float = 5e-4,
                 t0: float = 5.0, t1: float = 1.0,
                 lr: float = 0.005, num_epochs: int = 20):
        """
        Args:
            model (torch.nn.Module): model on which to make predictions
                The output of the model should be unnormalized class score.
                For example, last layer = CNConv or Linear.
            emb_layer_name (str, optional): name of the embedding layer
                If not specified, use the last but one layer by default.
            explain_graph (bool): whether the explanation is graph-level or node-level
            num_hops (int, optional): number of hops to consider
            coeff_size (float): size regularization to constrain the explanation size
            coeff_ent (float): entropy regularization to constrain the connectivity of explanation
            t0 (float): the temperature at the first epoch
            t1(float): the temperature at the final epoch
            lr (float): learning rate to train the explanation network
            num_epochs (int): number of epochs to train the explanation network
        """
        super().__init__(model, emb_layer_name)

        # Parameters for PGExplainer
        self.explain_graph = explain_graph
        self.num_hops = num_hops
        self.coeff_size = coeff_size
        self.coeff_ent = coeff_ent
        self.t0 = t0
        self.t1 = t1
        self.lr = lr
        self.num_epochs = num_epochs

        # Explanation model in PGExplainer
        if self.explain_graph:
            in_channels = 2 * self.emb_layer.out_channels
        else:
            in_channels = 3 * self.emb_layer.out_channels

        self.elayers = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(in_channels, 64),
                nn.ReLU()),
             nn.Linear(64, 1)])

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

    def __emb_to_edge_mask(self, emb: torch.Tensor,
                           x: torch.Tensor, edge_index: torch.Tensor,
                           node_idx: int = None,
                           forward_kwargs: dict = {},
                           tmp: float = 1.0, training: bool = False):
        """
        Compute the edge mask based on embedding.

        Returns:
            prob_with_mask (torch.Tensor): the predicted probability with edge_mask
            edge_mask (torch.Tensor): the mask for graph edges with values in [0, 1]
        """
        if not self.explain_graph and node_idx is None:
            raise ValueError('node_idx should be provided.')

        with torch.set_grad_enabled(training):
            # Concat relevant node embeddings
            U, V = edge_index  # edge (u, v), U = (u), V = (v)
            h1 = emb[U]
            h2 = emb[V]
            if self.explain_graph:
                h = torch.cat([h1, h2], dim=1)
            else:
                h3 = emb[node_idx].repeat(h1.shape[0], 1)
                h = torch.cat([h1, h2, h3], dim=1)

            # Calculate the edge weights and set the edge mask
            for elayer in self.elayers:
                h = elayer(h)
            h = h.squeeze()
            edge_weights = self.__concrete_sample(h, tmp, training)
            n = emb.shape[0]  # number of nodes
            mask_sparse = torch.sparse_coo_tensor(
                edge_index, edge_weights, (n, n))
            # Not scalable
            self.mask_sigmoid = mask_sparse.to_dense()
            # Undirected graph
            sym_mask = (self.mask_sigmoid + self.mask_sigmoid.transpose(0, 1)) / 2
            edge_mask = sym_mask[edge_index[0], edge_index[1]]
            self._set_masks(x, edge_index, edge_mask)

        # Compute the model prediction with edge mask
        prob_with_mask = self._predict(x, edge_index,
                                       forward_kwargs=forward_kwargs,
                                       return_type='prob')
        self._clear_masks()

        return prob_with_mask, edge_mask

    def train_explanation_model(self, dataset: Data, forward_kwargs: dict = {}):
        """
        Train the explanation model.
        """
        optimizer = torch.optim.Adam(self.elayers.parameters(), lr=self.lr)

        def loss_fn(prob: torch.Tensor, ori_pred: int):
            # Maximize the probability of predicting the label (cross entropy)
            loss = -torch.log(prob[ori_pred] + 1e-6)
            # Size regularization
            edge_mask = self.mask_sigmoid
            loss += self.coeff_size * torch.sum(edge_mask)
            # Element-wise entropy regularization
            # Low entropy implies the mask is close to binary
            edge_mask = edge_mask * 0.99 + 0.005
            entropy = - edge_mask * torch.log(edge_mask) \
                - (1 - edge_mask) * torch.log(1 - edge_mask)
            loss += self.coeff_ent * torch.mean(entropy)

            return loss

        if self.explain_graph:  # Explain graph-level predictions of multiple graphs
            # Get the embeddings and predicted labels
            with torch.no_grad():
                dataset_indices = list(range(len(dataset)))
                self.model.eval()
                emb_dict = {}
                ori_pred_dict = {}
                for gid in tqdm.tqdm(dataset_indices):
                    data = dataset[gid]
                    label = self._predict(data.x, data.edge_index,
                                          forward_kwargs=forward_kwargs)
                    emb = self._get_embedding(data.x, data.edge_index,
                                              forward_kwargs=forward_kwargs)
                    emb_dict[gid] = emb
                    ori_pred_dict[gid] = label

            # Train the mask generator
            duration = 0.0
            for epoch in range(self.num_epochs):
                loss = 0.0
                pred_list = []
                tmp = float(self.t0 * np.power(self.t1/self.t0, epoch/self.num_epochs))
                self.elayers.train()
                optimizer.zero_grad()
                tic = time.perf_counter()
                for gid in tqdm.tqdm(dataset_indices):
                    data = dataset[gid]
                    prob_with_mask, _ = self.__emb_to_edge_mask(
                        emb_dict[gid], data.x, data.edge_index,
                        forward_kwargs=forward_kwargs,
                        tmp=tmp, training=True)
                    loss_tmp = loss_fn(prob_with_mask.squeeze(), ori_pred_dict[gid])
                    loss_tmp.backward()
                    loss += loss_tmp.item()
                    pred_label = prob_with_mask.argmax(-1).item()
                    pred_list.append(pred_label)

                optimizer.step()
                duration += time.perf_counter() - tic
                print(f'Epoch: {epoch} | Loss: {loss}')

        else:  # Explain node-level predictions of a graph
            data = dataset
            # Get the predicted labels for training nodes
            with torch.no_grad():
                self.model.eval()
                explain_node_index_list = torch.where(data.train_mask)[0].tolist()
                pred_dict = {}
                label = self._predict(data.x, data.edge_index,
                                      forward_kwargs=forward_kwargs)
                for node_idx in tqdm.tqdm(explain_node_index_list):
                    pred_dict[node_idx] = label[node_idx]

            # Train the mask generator
            duration = 0.0
            for epoch in range(self.num_epochs):
                loss = 0.0
                optimizer.zero_grad()
                tmp = float(self.t0 * np.power(self.t1/self.t0, epoch/self.num_epochs))
                self.elayers.train()
                tic = time.perf_counter()
                for iter_idx, node_idx in tqdm.tqdm(enumerate(explain_node_index_list)):
                    subset, sub_edge_index, _, _ = \
                        k_hop_subgraph(node_idx, self.num_hops, data.edge_index,
                                        relabel_nodes=True, num_nodes=data.x.shape[0])

                    emb = self._get_embedding(data.x, data.edge_index,
                                                forward_kwargs=forward_kwargs)
                    new_node_index = int(torch.where(subset == node_idx)[0])
                    prob_with_mask, _ = self.__emb_to_edge_mask(
                        emb, data.x, data.edge_index, node_idx,
                        forward_kwargs=forward_kwargs,
                        tmp=tmp, training=True)
                    loss_tmp = loss_fn(prob_with_mask[new_node_index],
                                       pred_dict[node_idx])
                    loss_tmp.backward()
                    loss += loss_tmp.item()

                optimizer.step()
                duration += time.perf_counter() - tic
                print(f'Epoch: {epoch} | Loss: {loss/len(explain_node_index_list)}')

            print(f"training time is {duration:.5}s")

    def get_explanation_node(self, node_idx: int, x: torch.Tensor,
                             edge_index: torch.Tensor,
                             label: torch.Tensor = None,
                             forward_kwargs: dict = {}):
        """
        Explain a node prediction.

        Args:
            node_idx (int): index of the node to be explained
            x (torch.Tensor, [n x d]): node features
            edge_index (torch.Tensor, [2 x m]): edge index of the graph
            label (torch.Tensor, optional, [n x ...]): labels to explain
                If not provided, we use the output of the model.
            forward_kwargs (dict, optional): additional arguments to model.forward
                beyond x and edge_index

        Returns:
            exp (dict):
                exp['feature_imp'] (torch.Tensor, [d]): feature mask explanation
                exp['edge_imp'] (torch.Tensor, [m]): k-hop edge importance
                exp['node_imp'] (torch.Tensor, [n]): k-hop node importance
            khop_info (4-tuple of torch.Tensor):
                0. the nodes involved in the subgraph
                1. the filtered `edge_index`
                2. the mapping from node indices in `node_idx` to their new location
                3. the `edge_index` mask indicating which edges were preserved
        """
        if self.explain_graph:
            raise Exception('For graph-level explanations use `get_explanation_graph`.')

        label = self._predict(x, edge_index) if label is None else label

        exp = {k: None for k in EXP_TYPES}

        khop_info = k_hop_subgraph(node_idx, self.num_hops, edge_index,
                                   relabel_nodes=True, num_nodes=x.shape[0])

        emb = self._get_embedding(x, edge_index,
                                  forward_kwargs=forward_kwargs)
        _, edge_mask = self.__emb_to_edge_mask(emb, x, edge_index, node_idx,
                                               forward_kwargs=forward_kwargs,
                                               tmp=1.0, training=False)
        exp['edge_imp'] = edge_mask

        return exp, khop_info

    def get_explanation_graph(self, x: torch.Tensor, edge_index: torch.Tensor,
                              label: Optional[torch.Tensor] = None,
                              forward_kwargs: dict = {},
                              top_k: int = 10):
        """
        Explain a whole-graph prediction.

        Args:
            x (torch.Tensor, [n x d]): node features
            edge_index (torch.Tensor, [2 x m]): edge index of the graph
            label (torch.Tensor, optional, [n x ...]): labels to explain
                If not provided, we use the output of the model.
            forward_kwargs (dict, optional): additional arguments to model.forward
                beyond x and edge_index
            top_k (int): number of edges to include in the edge-importance explanation

        Returns:
            exp (dict):
                exp['feature_imp'] (torch.Tensor, [d]): feature mask explanation
                exp['edge_imp'] (torch.Tensor, [m]): k-hop edge importance
                exp['node_imp'] (torch.Tensor, [m]): k-hop node importance

        """
        if not self.explain_graph:
            raise Exception('For node-level explanations use `get_explanation_node`.')

        label = self._predict(x, edge_index,
                              forward_kwargs=forward_kwargs) if label is None else label

        exp = {k: None for k in EXP_TYPES}

        emb = self._get_embedding(x, edge_index,
                                  forward_kwargs=forward_kwargs)
        _, edge_mask = self.__emb_to_edge_mask(emb, x, edge_index,
                                               forward_kwargs=forward_kwargs,
                                               tmp=1.0, training=False)

        exp['edge_imp'] = edge_mask

        return exp

    def get_explanation_link(self):
        """
        Explain a link prediction.
        """
        raise NotImplementedError()

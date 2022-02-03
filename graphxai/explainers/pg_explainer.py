import numpy as np
import torch
import torch.nn as nn
import tqdm
import time

from typing import Optional
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GINConv

from graphxai.explainers._base import _BaseExplainer
from graphxai.utils.constants import EXP_TYPES
from graphxai.utils import Explanation, node_mask_from_edge_mask

device = "cuda" if torch.cuda.is_available() else "cpu"

class PGExplainer(_BaseExplainer):
    """
    PGExplainer

    Code adapted from DIG
    """
    def __init__(self, model: nn.Module, emb_layer_name: str = None,
                 explain_graph: bool = False,
                 coeff_size: float = 0.01, coeff_ent: float = 5e-4,
                 t0: float = 5.0, t1: float = 2.0,
                 lr: float = 0.003, max_epochs: int = 20, eps: float = 1e-3,
                 num_hops: int = None):
        """
        Args:
            model (torch.nn.Module): model on which to make predictions
                The output of the model should be unnormalized class score.
                For example, last layer = CNConv or Linear.
            emb_layer_name (str, optional): name of the embedding layer
                If not specified, use the last but one layer by default.
            explain_graph (bool): whether the explanation is graph-level or node-level
            coeff_size (float): size regularization to constrain the explanation size
            coeff_ent (float): entropy regularization to constrain the connectivity of explanation
            t0 (float): the temperature at the first epoch
            t1 (float): the temperature at the final epoch
            lr (float): learning rate to train the explanation network
            max_epochs (int): number of epochs to train the explanation network
            num_hops (int): number of hops to consider for node-level explanation
        """
        super().__init__(model, emb_layer_name)

        # Parameters for PGExplainer
        self.explain_graph = explain_graph
        self.coeff_size = coeff_size
        self.coeff_ent = coeff_ent
        self.t0 = t0
        self.t1 = t1
        self.lr = lr
        self.eps = eps
        self.max_epochs = max_epochs
        self.num_hops = self.L if num_hops is None else num_hops

        # Explanation model in PGExplainer

        mult = 2 if self.explain_graph else 3

        if isinstance(self.emb_layer, GCNConv):
            in_channels = mult * self.emb_layer.out_channels
        elif isinstance(self.emb_layer, GINConv):
            in_channels = mult * self.emb_layer.nn.out_features
        elif isinstance(self.emb_layer, torch.nn.Linear):
            in_channels = mult * self.emb_layer.out_features

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
            training == True: sigmoid((log_alpha + noise) / beta)
            training == False: sigmoid(log_alpha)
        """
        if training:
            random_noise = torch.rand(log_alpha.shape).to(device)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            gate_inputs = (random_noise + log_alpha) / beta
            gate_inputs = gate_inputs.sigmoid()
        else:
            gate_inputs = log_alpha.sigmoid()

        return gate_inputs

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
                h = elayer.to(device)(h)
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
            #print('edge_mask', edge_mask)
            self._set_masks(x, edge_index, edge_mask)

        # Compute the model prediction with edge mask
        # with torch.no_grad():
        #     tester = self.model(x, edge_index)
        #     print(tester)
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
                    data = dataset[gid].to(device)
                    pred_label = self._predict(data.x, data.edge_index,
                                               forward_kwargs=forward_kwargs)
                    emb = self._get_embedding(data.x, data.edge_index,
                                              forward_kwargs=forward_kwargs)
                    ori_pred_dict[gid] = pred_label

            # Train the mask generator
            duration = 0.0
            last_loss = 0.0
            for epoch in range(self.max_epochs):
                loss = 0.0
                pred_list = []
                tmp = float(self.t0 * np.power(self.t1/self.t0, epoch/self.max_epochs))
                self.elayers.train()
                optimizer.zero_grad()
                tic = time.perf_counter()
                for gid in tqdm.tqdm(dataset_indices):
                    data = dataset[gid].to(device)
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
                if abs(loss - last_loss) < self.eps:
                    break
                last_loss = loss

        else:  # Explain node-level predictions of a graph
            data = dataset.to(device)
            # Get the predicted labels for training nodes
            with torch.no_grad():
                self.model.eval()
                explain_node_index_list = torch.where(data.train_mask)[0].tolist()
                # pred_dict = {}
                label = self._predict(data.x, data.edge_index,
                                      forward_kwargs=forward_kwargs)
                pred_dict = dict(zip(explain_node_index_list, label[explain_node_index_list]))
                # for node_idx in tqdm.tqdm(explain_node_index_list):
                #     pred_dict[node_idx] = label[node_idx]

            # Train the mask generator
            duration = 0.0
            last_loss = 0.0
            for epoch in range(self.max_epochs):
                loss = 0.0
                optimizer.zero_grad()
                tmp = float(self.t0 * np.power(self.t1/self.t0, epoch/self.max_epochs))
                self.elayers.train()
                tic = time.perf_counter()
                emb = self._get_embedding(data.x, data.edge_index,
                                          forward_kwargs=forward_kwargs)
                X = data.x
                EIDX = data.edge_index
                for iter_idx, node_idx in tqdm.tqdm(enumerate(explain_node_index_list)):
                    import ipdb; ipdb.set_trace()
                    subset, _, _, _ = \
                        k_hop_subgraph(node_idx, self.num_hops, data.edge_index,
                                       relabel_nodes=True, num_nodes=data.x.shape[0])
                    new_node_index = int(torch.where(subset == node_idx)[0])
                    # print('data.x', data.x)
                    # print('data.edge_index', data.edge_index)
                    # print('forward_kwargs', forward_kwargs)
                    prob_with_mask, _ = self.__emb_to_edge_mask(
                        emb, 
                        x = X, 
                        edge_index = EIDX, 
                        node_idx = node_idx,
                        forward_kwargs=forward_kwargs,
                        tmp=tmp, 
                        training=True)
                    loss_tmp = loss_fn(prob_with_mask[new_node_index],
                                       pred_dict[node_idx])
                    loss_tmp.backward()
                    loss += loss_tmp.item()

                optimizer.step()
                duration += time.perf_counter() - tic
#                print(f'Epoch: {epoch} | Loss: {loss/len(explain_node_index_list)}')
                if abs(loss - last_loss) < self.eps:
                    break
                last_loss = loss

            print(f"training time is {duration:.5}s")

    def get_explanation_node(self, node_idx: int, x: torch.Tensor,
                             edge_index: torch.Tensor, label: torch.Tensor = None,
                             y = None,
                             forward_kwargs: dict = {}, **_):
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
        khop_info = _, _, _, sub_edge_mask = \
            k_hop_subgraph(node_idx, self.num_hops, edge_index,
                           relabel_nodes=False, num_nodes=x.shape[0])
        emb = self._get_embedding(x, edge_index, forward_kwargs=forward_kwargs)
        _, edge_mask = self.__emb_to_edge_mask(
            emb, x, edge_index, node_idx, forward_kwargs=forward_kwargs,
            tmp=2, training=False)
        edge_imp = edge_mask[sub_edge_mask]

        exp = Explanation(
            node_imp = node_mask_from_edge_mask(khop_info[0], khop_info[1], edge_imp.bool()),
            edge_imp = edge_imp,
            node_idx = node_idx
        )

        exp.set_enclosing_subgraph(khop_info)

        return exp

    def get_explanation_graph(self, x: torch.Tensor, edge_index: torch.Tensor,
                              label: Optional[torch.Tensor] = None,
                              y = None,
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

        #exp = {k: None for k in EXP_TYPES}

        with torch.no_grad():
            emb = self._get_embedding(x, edge_index,
                                      forward_kwargs=forward_kwargs)
            _, edge_mask = self.__emb_to_edge_mask(
                emb, x, edge_index, forward_kwargs=forward_kwargs,
                tmp=1.0, training=False)

        #exp['edge_imp'] = edge_mask

        exp = Explanation(
            edge_imp = edge_mask
        )

        return exp

    def get_explanation_link(self):
        """
        Explain a link prediction.
        """
        raise NotImplementedError()

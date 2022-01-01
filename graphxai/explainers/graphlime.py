import numpy as np
import torch
import torch.nn as nn

from sklearn.linear_model import LassoLars
from torch_geometric.utils import k_hop_subgraph

from graphxai.explainers._base import _BaseExplainer
from graphxai.utils import Explanation


class GraphLIME(_BaseExplainer):
    """
    GraphLIME: node only

    (Code adapted from https://github.com/WilliamCCHuang/GraphLIME)
    """
    def __init__(self, model: nn.Module,
                 rho: float = 0.1, *_):
        """
        Args:
            model (torch.nn.Module): model on which to make predictions
                The output of the model should be unnormalized class score.
                For example, last layer = GCNConv or Linear.
            rho (float): regularization strength
        """
        super().__init__(model)
        self.rho = rho

    def _compute_kernel(self, x: np.ndarray, reduce: bool):
        """
        Compute the RBF kernel matrix.

        Args:
            x (np.ndarray, [n x d]): node feautres
            reduce (bool): whether to reduce the last axis

        Returns:
            K (np.ndarray): RBF kernel matrix
                If reduce == False, K is [n x n x d].
                If reduce == True, the last axis is reduced by sum and K is [n x n x 1].
        """
        n, d = x.shape
        dist = x.reshape(1, n, d) - x.reshape(n, 1, d)  # pairwise distance [n x n x d]
        dist = dist ** 2

        if reduce:
            dist = np.sum(dist, axis=-1, keepdims=True)

        K = np.exp(-dist / (2 * np.sqrt(d) ** 2 * 0.1 + 1e-10))

        return K

    def _standardize_kernel(self, K: np.ndarray):
        """
        Standardize the kernel matrix.

        Args:
            K (np.ndarray, [n x n x d]): kernel matrix

        Returns:
            K_bar (np.ndarray, [n x n x d]): centralized and normalized kernel matrix
        """
        # Centralize
        G = K - np.mean(K, axis=0, keepdims=True)
        G = G - np.mean(G, axis=1, keepdims=True)
        # Normalize
        G = G / (np.linalg.norm(G, ord='fro', axis=(0, 1), keepdims=True) + 1e-10)
        return G

    def get_explanation_node(self, node_idx: int, edge_index: torch.Tensor,
                             x: torch.Tensor, num_hops: int = None,
                             forward_kwargs: dict = {}):
        """
        Explain a node prediction.

        Args:
            node_idx (int): index of the node to be explained
            edge_index (torch.Tensor, [2 x m]): edge index of the graph
            x (torch.Tensor, [N x d]): node features
            label (torch.Tensor, [N x ...]): labels to explain
            num_hops (int): number of hops to consider

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
        num_hops = num_hops if num_hops is not None else self.L

        y = self._predict(x, edge_index, return_type='prob',
                          forward_kwargs=forward_kwargs)
        khop_info = subset, sub_edge_index, mapping, _ = \
            k_hop_subgraph(node_idx, num_hops, edge_index,
                           relabel_nodes=True, num_nodes=x.shape[0])
        sub_x = x[subset].detach().cpu().numpy()  # [n x d]
        sub_y = y[subset].detach().cpu().numpy()  # [n x 1]

        #exp = {k: None for k in EXP_TYPES}

        n, d = sub_x.shape

        K = self._compute_kernel(sub_x, reduce=False)
        L = self._compute_kernel(sub_y, reduce=True)

        K_bar = self._standardize_kernel(K)  # [n x n x d]
        L_bar = self._standardize_kernel(L)  # [n x n x 1]

        K_bar = K_bar.reshape(n ** 2, d)
        L_bar = L_bar.reshape(n ** 2)

        solver = LassoLars(self.rho, fit_intercept=False, normalize=False, positive=True)
        solver.fit(K_bar * n, L_bar * n)

        feat_imp = torch.from_numpy(solver.coef_)

        exp = Explanation(feature_imp=feat_imp)
        exp.set_enclosing_subgraph(khop_info)

        return exp

    def get_explanation_graph(self, edge_index: torch.Tensor,
                              x: torch.Tensor, label: torch.Tensor,
                              forward_kwargs: dict = {}):
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
        raise Exception('GraphLIME cannot provide explanation \
                        for graph-level prediction.')

    def get_explanation_link(self):
        """
        Explain a link prediction.
        """
        raise NotImplementedError()

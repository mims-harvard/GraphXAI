import numpy as np
import pandas as pd
import torch

from typing import List
from torch_geometric.utils import k_hop_subgraph
from pgmpy.estimators.CITests import chi_square
from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination

from graphxai.explainers._base import _BaseExplainer
from graphxai.utils.perturb import perturb_node_features
from graphxai.utils.constants import EXP_TYPES
from .utils import chi_square_pgm, generalize_target, generalize_others


class PGMExplainer(_BaseExplainer):
    """
    PGMExplainer

    Reference: https://github.com/vunhatminh/PGMExplainer
    """
    def __init__(self, model: torch.nn.Module,
                 num_samples: int = 100, perturb_prob: float = 0.5,
                 p_threshold: float = 0.05, pred_diff_threshold: float = 0.1):
        """
        Args:
            model (torch.nn.Module): model on which to make predictions
                The output of the model should be unnormalized class score.
                For example, last layer = CNConv or Linear.
            num_samples (int): number of perturbed graphs to generate
            perturb_prob (float): the probability that a node's features are perturbed
            p_threshold (float): the threshold for chi-square independence test
            pred_diff_threshold (float): the threshold for the difference of
                predicted class probability
        """
        super().__init__(model)
        self.num_samples = num_samples
        self.perturb_prob = perturb_prob
        self.p_threshold = p_threshold
        self.pred_diff_threshold = pred_diff_threshold


    def __search_Markov_blanket(self, df: pd.DataFrame,
                                target_node: str, nodes: List[str]):
        """
        Search the Markov blanket of target_node inside nodes.

        Might run forever.
        """
        MB = nodes
        num_iters = 0
        while True:
            num_iters += 1
            print(num_iters)
            count = 0
            for node in nodes:
                evidence = MB.copy()
                evidence.remove(node)
                _, p = chi_square_pgm(target_node, node, evidence,
                                      df[nodes + [target_node]])
                if p > self.p_threshold:
                    MB.remove(node)
                    count = 0
                else:
                    count += 1
                    if count == len(MB):  # No nodes inside MB can be removed.
                        return MB

    def get_explanation_node(self, node_idx: int, x: torch.Tensor,
                             edge_index: torch.Tensor, num_hops: int = 3,
                             forward_kwargs: dict = {},
                             top_k_nodes: int = None, no_child: bool = True):
        """
        Explain a node prediction.

        Args:
            node_idx (int): index of the node to be explained
            edge_index (torch.Tensor, [2 x m]): edge index of the graph
            x (torch.Tensor, [n x d]): node features
            num_hops (int, optional): number of hops to consider
            forward_kwargs (dict, optional): additional arguments to model.forward
                beyond x and edge_index
            top_k_nodes (int, optional): number of nodes to include in the PGM.
                If not provided, keep all the nodes given by the chi-square test.

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
        num_hops = self.L if num_hops is None else num_hops

        exp = {k: None for k in EXP_TYPES}

        khop_info = subset, sub_edge_index, _, _ = \
            k_hop_subgraph(node_idx, num_hops, edge_index,
                           relabel_nodes=True, num_nodes=x.shape[0])
        sub_x = x[subset]

        pred_prob = self._predict(x, edge_index, return_type='prob',
                                  forward_kwargs=forward_kwargs)

        # Only consider the neighbors
        pred_prob = pred_prob[subset]
        pred_label = pred_prob.argmax(dim=1)

        sub_n = sub_x.shape[0]
        sample_pert = np.zeros((self.num_samples, sub_n), dtype=int)
        sample_pred_diff = np.zeros((self.num_samples, sub_n), dtype=int)
        for iter_idx in range(self.num_samples):
            # Perturb the features of randomly selected nodes
            x_pert, pert_mask = \
                perturb_node_features(sub_x, perturb_prob=self.perturb_prob)
            # pert_mask stores whether the node is perturbed
            pert_mask = pert_mask.numpy().astype(int)
            sample_pert[iter_idx, :] = pert_mask

            # Compute the prediction after perturbation of features
            pred_prob_pert = self._predict(x_pert, sub_edge_index, return_type='prob',
                                           forward_kwargs=forward_kwargs)
            # pred_diff_mask stores whether the pert causes significant difference
            pred_diff = pred_prob[torch.arange(sub_n), pred_label] - \
                pred_prob_pert[torch.arange(sub_n), pred_label]
            pred_diff_mask = (pred_diff > self.pred_diff_threshold).numpy().astype(int)
            sample_pred_diff[iter_idx, :] = pred_diff_mask

        # For each sample, compute a combined value: 10 * pert + pred_diff + 1
        # Here pert = 1 if the node's features have been perturbed
        # and pred_diff = 1 if the node's prediction has changed significantly
        sample_comb = sample_pert * 10 + sample_pred_diff + 1

        # Make data frame for chi-square conditional independence test
        neighbors = subset.tolist()
        df = pd.DataFrame(sample_comb)
        ind_ori_to_sub = dict(zip(neighbors, list(df.columns)))
        ind_sub_to_ori = dict(zip(list(df.columns), neighbors))

        # Compute p-values for each neighbor
        p_values = []
        dependent_neighbors = []
        for node in neighbors:
            _, p, _ = chi_square(ind_ori_to_sub[node], ind_ori_to_sub[node_idx], [],
                                 df, boolean=False)
            p_values.append(p)
            if p < self.p_threshold:
                dependent_neighbors.append(node)
        pgm_stats = dict(zip(neighbors, p_values))

        sub_nodes = []
        if top_k_nodes is None:
            sub_nodes = dependent_neighbors
        else:
            top_k_nodes = min(top_k_nodes, len(neighbors) - 1)
            ind_top_k = np.argpartition(p_values, top_k_nodes)[:top_k_nodes]
            sub_nodes = [ind_sub_to_ori[node] for node in ind_top_k]

        sub_nodes = [str(int(node)) for node in sub_nodes]
        target_node = str(int(node_idx))
        sub_nodes_no_target = [node for node in sub_nodes if node != target_node]
        df = df.rename(columns=ind_sub_to_ori)
        df.columns = df.columns.astype(str)

        # Get the Markov blanket
        MB = self.__search_Markov_blanket(df, target_node, sub_nodes_no_target.copy())

        # Perform structural learning
        if no_child:
            est = HillClimbSearch(df[sub_nodes_no_target], scoring_method=BicScore(df))
            pgm_no_target = est.estimate()
            for node in MB:
                if node != target_node:
                    pgm_no_target.add_edge(node, target_node)

            # Create the PGM
            pgm_explanation = BayesianModel()
            for node in pgm_no_target.nodes():
                pgm_explanation.add_node(node)
            for edge in pgm_no_target.edges():
                pgm_explanation.add_edge(edge[0], edge[1])

            # Fit the PGM
            df_ex = df[sub_nodes].copy()
            df_ex[target_node] = df[target_node].apply(generalize_target)
            for node in sub_nodes_no_target:
                df_ex[node] = df[node].apply(generalize_others)
            pgm_explanation.fit(df_ex)
        else:
            df_ex = df[sub_nodes].copy()
            df_ex[target_node] = df[target_node].apply(generalize_target)
            for node in sub_nodes_no_target:
                df_ex[node] = df[node].apply(generalize_others)

            est = HillClimbSearch(df_ex, scoring_method=BicScore(df_ex))
            pgm_w_target_explanation = est.estimate()

            # Create the PGM    
            pgm_explanation = BayesianModel()
            for node in pgm_w_target_explanation.nodes():
                pgm_explanation.add_node(node)
            for edge in pgm_w_target_explanation.edges():
                pgm_explanation.add_edge(edge[0],edge[1])

            # Fit the PGM
            df_ex = df[sub_nodes].copy()
            df_ex[target_node] = df[target_node].apply(generalize_target)
            for node in sub_nodes_no_target:
                df_ex[node] = df[node].apply(generalize_others)
            pgm_explanation.fit(df_ex)

        return pgm_explanation

    def get_explanation_graph(self, x: torch.Tensor,
                              edge_index: torch.Tensor, label: torch.Tensor,
                              forward_kwargs: dict = None, explain_feature: bool = False):
        """
        Explain a whole-graph prediction.

        Args:
            edge_index (torch.Tensor, [2 x m]): edge index of the graph
            x (torch.Tensor, [n x d]): node features
            label (torch.Tensor, [n x ...]): labels to explain
            forward_kwargs (dict, optional): additional arguments to model.forward
                beyond x and edge_index
            explain_feature (bool): whether to explain the feature or not

        Returns:
            exp (dict):
                exp['feature_imp'] (torch.Tensor, [d]): feature mask explanation
                exp['edge_imp'] (torch.Tensor, [m]): k-hop edge importance
                exp['node_imp'] (torch.Tensor, [m]): k-hop node importance
        """
        raise Exception('GNNExplainer does not support graph-level explanation.') 

    def get_explanation_link(self):
        """
        Explain a link prediction.
        """
        raise NotImplementedError()

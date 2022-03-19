import numpy as np
import pandas as pd
import torch

from typing import List
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data
from pgmpy.estimators.CITests import chi_square
from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.models import BayesianNetwork

from graphxai.explainers._base import _BaseExplainer
from graphxai.utils.perturb import PGM_perturb_node_features
from graphxai.utils import Explanation
from .utils import chi_square_pgm, generalize_target, generalize_others

device = "cuda" if torch.cuda.is_available() else "cpu"

class PGMExplainer(_BaseExplainer):
    """
    PGMExplainer

    Reference: https://github.com/vunhatminh/PGMExplainer
    """
    def __init__(self, model: torch.nn.Module,
                 explain_graph: bool, num_samples: int = None,
                 perturb_mode: str = None, perturb_prob: float = 0.5,
                 p_threshold: float = 0.05, pred_diff_threshold: float = 0.1):
        """
        Args:
            model (torch.nn.Module): model on which to make predictions
                The output of the model should be unnormalized class score.
                For example, last layer = CNConv or Linear.
            explain_graph (bool): whether to explain graph classification model
            num_samples (int): number of perturbed graphs to generate
            perturb_mode (str):
                'scale': randomly scale (0-2) the continuous dim
                'gaussian': add a Gaussian noise to the continuous dim
                'uniform': add a uniform noise to the continuous dim
                'mean': set the continuous dims to their mean value
            perturb_prob (float): the probability that a node's features are perturbed
            p_threshold (float): the threshold for chi-square independence test
            pred_diff_threshold (float): the threshold for the difference of
                predicted class probability
        """
        super().__init__(model)
        self.explain_graph = explain_graph

        if num_samples is None:
            self.num_samples = 10 if explain_graph else 100
        else:
            self.num_samples = num_samples

        if num_samples is None:
            self.num_samples = 10 if explain_graph else 100
        else:
            self.num_samples = num_samples

        if perturb_mode is None:
            self.perturb_mode = 'mean' if explain_graph else 'scale'
        else:
            if perturb_mode not in ['scale', 'gaussian', 'uniform', 'mean']:
                raise ValueError("perturb_mode must be one of ['scale', 'gaussian', 'uniform', 'mean']")
            self.perturb_mode = perturb_mode

        self.perturb_prob = perturb_prob
        self.p_threshold = p_threshold
        self.pred_diff_threshold = pred_diff_threshold

    def __search_Markov_blanket(self, df: pd.DataFrame,
                                target_node: str, nodes: List[str]):
        """
        Search the Markov blanket of target_node inside nodes.
        """
        MB = nodes
        i = 0
        while True:

            i += 1
            if i > 1000:
                return MB

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

    def __get_perturb_diff(self, x: torch.Tensor, edge_index: torch.Tensor,
                           num_samples: int, subset = None, forward_kwargs: dict = {}):
        """
        Get a [num_samples x n x 2] int array, where the last dim stores
        0. whether a node is perturbed
        1. whether prediction at a node varies significantly

        Args:
            x (torch.Tensor, [n x d]): node features
            edge_index (torch.Tensor, [2 x m]): edge index of the graph
            subset ([k]): subset of nodes that may get perturbed
            num_samples (int): number of perturbed graphs to generate
            forward_kwargs (dict, optional): additional arguments to model.forward
                beyond x and edge_index

        Returns:
            sample (np.ndarray, [num_samples x n x 2])
        """
        pred_prob = self._predict(x, edge_index, return_type='prob',
                                  forward_kwargs=forward_kwargs)
        pred_label = pred_prob.argmax(dim=-1)

        if self.explain_graph:  # graph-level explanation
            n = x.shape[0]
            sample = np.zeros((num_samples, n+1), dtype=int)
            sample_pred_diff = np.zeros(num_samples)

            for iter_idx in range(num_samples):
                # Perturb the features of randomly selected nodes
                x_pert, pert_mask = \
                    PGM_perturb_node_features(x, perturb_prob=self.perturb_prob,
                                          perturb_mode=self.perturb_mode)
                # pert_mask stores whether a node is perturbed
                pert_mask = pert_mask.numpy().astype(int)
                sample[iter_idx, :n] = pert_mask

                # Compute the prediction after perturbation of features
                pred_prob_pert = self._predict(x_pert, edge_index, return_type='prob',
                                               forward_kwargs=forward_kwargs)
                # pred_diff_mask stores whether the pert causes significant difference
                pred_diff = pred_prob[pred_label] - pred_prob_pert[pred_label]
                sample_pred_diff[iter_idx] = pred_diff

            # Set the pred_diff_mask of the top 1/8 samples to 1
            top = num_samples // 8
            top_indices = np.argsort(sample_pred_diff)[-top:]
            sample[top_indices, n] = 1

        else:  # node-level explanation
            sub_x = x[subset]
            sub_n = sub_x.shape[0]
            sample = np.zeros((num_samples, sub_n, 2), dtype=int)

            for iter_idx in range(num_samples):
                # Perturb the features of randomly selected nodes
                sub_x_pert, pert_mask = \
                    PGM_perturb_node_features(sub_x, perturb_prob=self.perturb_prob,
                                          perturb_mode=self.perturb_mode)
                x_pert = x.clone()
                x_pert[subset] = sub_x_pert
                # pert_mask stores whether a node is perturbed
                pert_mask = pert_mask.numpy().astype(int)
                sample[iter_idx, :sub_n, 0] = pert_mask

                # Compute the prediction after perturbation of features
                pred_prob_pert = self._predict(x_pert, edge_index, return_type='prob',
                                            forward_kwargs=forward_kwargs)
                # pred_diff_mask stores whether the pert causes significant difference
                pred_diff = pred_prob[subset, pred_label[subset]] - \
                    pred_prob_pert[subset, pred_label[subset]]
                pred_diff_mask = pred_diff > self.pred_diff_threshold
                pred_diff_mask = pred_diff_mask.cpu().numpy().astype(int)
                sample[iter_idx, :sub_n, 1] = pred_diff_mask

        return sample

    def get_explanation_node(self, node_idx: int, x: torch.Tensor,
                             edge_index: torch.Tensor, y = None, num_hops: int = None,
                             forward_kwargs: dict = {},
                             top_k_nodes: int = None, no_child: bool = True):
        """
        Explain a node prediction.

        Args:
            node_idx (int): index of the node to be explained
            x (torch.Tensor, [n x d]): node features
            edge_index (torch.Tensor, [2 x m]): edge index of the graph
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
        if self.explain_graph:
            raise Exception('For graph-level explanations use `get_explanation_graph`.')

        num_hops = self.L if num_hops is None else num_hops

        khop_info = subset, edge_index, _, _ = \
            k_hop_subgraph(node_idx, num_hops, edge_index,
                           relabel_nodes=True, num_nodes=x.shape[0])
        n = x.shape[0]

        # Get pert, diff sample
        sample = self.__get_perturb_diff(x, edge_index, num_samples=self.num_samples,
                                         subset=subset, forward_kwargs=forward_kwargs)

        # For each sample, compute a combined value: 10 * pert + pred_diff + 1
        # Here pert = 1 if the node's features have been perturbed
        # and pred_diff = 1 if the node's prediction has changed significantly
        sample_comb = sample[:, :, 0] * 10 + sample[:, :, 1] + 1

        # Make data frame for chi-square conditional independence test
        neighbors = subset.tolist()
        df = pd.DataFrame(sample_comb)
        ind_sub_to_ori = dict(zip(list(df.columns), neighbors))
        ind_ori_to_sub = dict(zip(neighbors, list(df.columns)))

        # Compute p-values for each neighbor
        p_values = []
        dependent_neighbors = []
        for node in neighbors:
            _, p, _ = chi_square(ind_ori_to_sub[node], ind_ori_to_sub[node_idx], [], df, boolean=False)
            p_values.append(p)
            if p < self.p_threshold:
                dependent_neighbors.append(node)

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
            # est = HillClimbSearch(df[sub_nodes_no_target], scoring_method=BicScore(df))
            # pgm_no_target = est.estimate()
            est = HillClimbSearch(df[sub_nodes_no_target])
            pgm_no_target = est.estimate(scoring_method=BicScore(df))
            for node in MB:
                if node != target_node:
                    pgm_no_target.add_edge(node, target_node)

            # Create the PGM
            #pgm = BayesianModel()
            pgm = BayesianNetwork()
            for node in pgm_no_target.nodes():
                pgm.add_node(node)
            for edge in pgm_no_target.edges():
                pgm.add_edge(edge[0], edge[1])

            # Fit the PGM
            df_ex = df[sub_nodes].copy()
            df_ex[target_node] = df[target_node].apply(generalize_target)
            for node in sub_nodes_no_target:
                df_ex[node] = df[node].apply(generalize_others)
            pgm.fit(df_ex)
        else:
            df_ex = df[sub_nodes].copy()
            df_ex[target_node] = df[target_node].apply(generalize_target)
            for node in sub_nodes_no_target:
                df_ex[node] = df[node].apply(generalize_others)

            est = HillClimbSearch(df_ex, scoring_method=BicScore(df_ex))
            pgm_w_target_explanation = est.estimate()

            # Create the PGM    
            #pgm = BayesianModel()
            pgm = BayesianNetwork()
            for node in pgm_w_target_explanation.nodes():
                pgm.add_node(node)
            for edge in pgm_w_target_explanation.edges():
                pgm.add_edge(edge[0],edge[1])

            # Fit the PGM
            df_ex = df[sub_nodes].copy()
            df_ex[target_node] = df[target_node].apply(generalize_target)
            for node in sub_nodes_no_target:
                df_ex[node] = df[node].apply(generalize_others)
            pgm.fit(df_ex)

        pgm_nodes = [int(node) for node in pgm.nodes()]
        #print('pgm nodes', pgm_nodes)
        node_imp = torch.zeros(n)
        node_imp[pgm_nodes] = 1

        exp = Explanation(
            node_imp = node_imp[subset],
            node_idx = node_idx
        )

        # Store PGM in the Explanation
        exp.pgm = pgm

        exp.set_enclosing_subgraph(khop_info)

        return exp

    def get_explanation_graph(self, x: torch.Tensor, edge_index: torch.Tensor,
                              y = None,
                              forward_kwargs: dict = {},
                              top_k_nodes: int = None):
        """
        Explain a whole-graph prediction.

        Args:
            edge_index (torch.Tensor, [2 x m]): edge index of the graph
            x (torch.Tensor, [n x d]): node features
            top_k_nodes (int, optional): number of nodes to include in the PGM.
                If not provided, keep all the nodes given by the chi-square test.
            forward_kwargs (dict, optional): additional arguments to model.forward
                beyond x and edge_index

        Returns:
            exp (dict):
                exp['feature_imp'] (torch.Tensor, [d]): feature mask explanation
                exp['edge_imp'] (torch.Tensor, [m]): k-hop edge importance
                exp['node_imp'] (torch.Tensor, [m]): k-hop node importance
        """
        if not self.explain_graph:
            raise Exception('For node-level explanations use `get_explanation_node`.')

        n = x.shape[0]
        top_k_nodes = n // 20 if top_k_nodes is None else min(top_k_nodes, n-1)

        target = n  # pred_diff_mask is stored at column n

        # Round 1: all nodes may be perturbed, half the number of samples

        # Get pert, diff sample
        sample = self.__get_perturb_diff(x, edge_index,
                                         num_samples=self.num_samples//2,
                                         forward_kwargs=forward_kwargs)
        df = pd.DataFrame(sample)

        # Compute p-values and pick the candidate nodes to perturb
        p_values = []
        for node in range(n):
            _, p, _ = chi_square(node, target, [], df, boolean=False)
            p_values.append(p)
        num_candidates = min(top_k_nodes*4, n-1)
        candidate_nodes = np.argpartition(p_values, num_candidates)[0:num_candidates]

        # Round 2: only candidate nodes may be perturbed, the full number of samples

        # Get pert, diff sample
        sample = self.__get_perturb_diff(x, edge_index, num_samples=self.num_samples,
                                         subset=candidate_nodes,
                                         forward_kwargs=forward_kwargs)
        df = pd.DataFrame(sample)

        # Compute p-values and pick the nodes for PGM explanation
        p_values = []
        dependent_nodes = []
        for node in range(n):
            _, p,_ = chi_square(node, target, [], df, boolean=False)
            p_values.append(p)
            if p < self.p_threshold:
                dependent_nodes.append(node)

        ind_top_k = np.argpartition(p_values, top_k_nodes)[:top_k_nodes]
        node_imp = torch.zeros(n)
        node_imp[ind_top_k] = 1

        exp = Explanation(
            node_imp = node_imp
        )
    
        exp.set_whole_graph(Data(x=x, edge_index = edge_index))
        return exp

    def get_explanation_link(self):
        """
        Explain a link prediction.
        """
        raise NotImplementedError()

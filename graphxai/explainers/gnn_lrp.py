import torch
from torch import Tensor
import torch.nn as nn
import copy
import numpy as np
from collections import Iterable
from typing import Callable
from torch_geometric.utils.loop import add_self_loops
#from ..models.utils import subgraph
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.utils.num_nodes import maybe_num_nodes
#from ..models.models import GraphSequential
from .utils.base_explainer import WalkBase
#from .utils.base_explainer_org import WalkBase

EPS = 1e-15

def all_incoming_edges_w_node(edge_index, node_idx, row = 0):
    '''Gets all incoming edges to a given node provided row index'''
    return (edge_index[row,:] == node_idx).nonzero(as_tuple=True)[0].tolist()

class GraphSequential(nn.Sequential):

    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, *input) -> Tensor:
        for module in self:
            if isinstance(input, tuple):
                # print('Input', input)
                # print('len input', len(input))
                input = module(*input)
            else:
                input = module(input)
        return input

class GNN_LRP(WalkBase):
    r"""
    An implementation of GNN-LRP in
    `Higher-Order Explanations of Graph Neural Networks via Relevant Walks <https://arxiv.org/abs/2006.03589>`_.

    Args:
        model (torch.nn.Module): The target model prepared to explain.
        explain_graph (bool, optional): Whether to explain graph classification model.
            (default: :obj:`False`)

    .. note::
            For node classification model, the :attr:`explain_graph` flag is False.
            GNN-LRP is very model dependent. Please be sure you know how to modify it for different models.
            For an example, see `benchmarks/xgraph
            <https://github.com/divelab/DIG/tree/dig/benchmarks/xgraph>`_.

    .. note:: 
            Currently only supports models with GCN and/or GIN layers.

    """

    def __init__(self, model: nn.Module, explain_graph=False):
        super().__init__(model=model, explain_graph=explain_graph)

    def get_explanation_node(self, 
            x: Tensor, 
            edge_index: Tensor,
            node_idx: int,
            num_classes: int,
            forward_args: tuple = None,
            get_edge_scores: bool = True,
            edge_aggregator: Callable[[list], float] = np.sum,
            **kwargs
        ):
        '''
        Get explanation for computational graph around one node.

        .. note:: 
            `edge_aggregator` must take one argument, a list, and return one scalar (float).

        Args:
            x (Tensor): Graph input features.
            edge_index (Tensor): Input edge_index of graph.
            node_idx (int): Index of node for which to generate prediction and explanation 
                for corresponding prediction.
            num_classes (int): Number of classes in model.
            forward_args (tuple, optional): Any additional inputs to model.forward method (other 
                than x and edge_index). (default: :obj:`None`)
            get_edge_scores (bool, optional): If true, returns edge scores as combined by
                `edge_aggregator` function. (default: :obj:`True`)
            edge_aggregator (function, optional): Function to combine scores from multiple walks
                across one edge. Argument only has effect if `get_edge_scores == True`.
                (default: :obj:`numpy.sum`)

        :rtype: 
            If `get_edge_scores == True`, (:obj:`list`, :obj:`tuple`). Return is list of aggregated
                explanation values for each edge in subgraph. First dimension of list corresponds to label 
                that those values explain, i.e. `len(exp) == num_classes` with 
                `len(exp[i]) == # nodes in subgraph` for all i. Second return in tuple is `khop_info`,
                the information about the computational graph around node `node_idx` as is returned
                by `torch_geometric.utils.k_hop_subgraph`. Index in list corresponds exactly to 
                index of edges in `khop_info` (`khop_info[1]`), the second return in the tuple.
            If `get_edge_scores == False`, (:obj:`dict`, :obj:`tuple`). Return is dict of walks, with
                keys `'ids'` and `'scores'`, corresponding to walks as denoted by edge indices and scores
                of those corresponding walks, respectively. Second return is still `khop_info`.
        '''

        super().forward(x, edge_index, **kwargs)
        self.model.eval()

        walk_steps, fc_steps = self.extract_step(x, edge_index, detach=False, split_fc=True, 
            forward_args = forward_args)

        edge_index_with_loop, _ = add_self_loops(edge_index, num_nodes=self.num_nodes)


        walk_indices_list = torch.tensor(
            self.walks_pick(edge_index_with_loop.cpu(), list(range(edge_index_with_loop.shape[1])),
                            num_layers=self.num_layers), device=self.device)

        # Get subgraph of nodes in computational graph:
        khop_info  = k_hop_subgraph(
            node_idx, self.__num_hops__, edge_index_with_loop, relabel_nodes=True,
            num_nodes=None, flow=self.__flow__())

        self.hard_edge_mask = khop_info[-1]

        # walk indices list mask
        edge2node_idx = edge_index_with_loop[1] == node_idx
        walk_indices_list_mask = edge2node_idx[walk_indices_list[:, -1]]
        walk_indices_list = walk_indices_list[walk_indices_list_mask]


        def compute_walk_score():

            # hyper-parameter gamma
            epsilon = 1e-30   # prevent from zero division
            gamma = [2, 1, 1] # Hard-coded from DIG
            # TODO: Make this a hyperparameter

            # --- record original weights of GNN ---
            ori_gnn_weights = []
            gnn_gamma_modules = []
            clear_probe = x
            for i, walk_step in enumerate(walk_steps):
                modules = walk_step['module']
                gamma_ = gamma[i] if i <= 1 else 1
                if hasattr(modules[0], 'nn'):
                    clear_probe = modules[0](clear_probe, edge_index, probe=False)
                    # clear nodes that are not created by user

                gamma_module = copy.deepcopy(modules[0])
                # Creates modified version of the module

                if hasattr(modules[0], 'nn'):
                    for j, fc_step in enumerate(gamma_module.fc_steps):
                        fc_modules = fc_step['module']
                        if hasattr(fc_modules[0], 'weight'):
                            ori_fc_weight = fc_modules[0].weight.data
                            fc_modules[0].weight.data = ori_fc_weight + gamma_ * ori_fc_weight
                else:
                    ori_gnn_weights.append(modules[0].weight.data)

                    # (.)^ = (.) + \gamma * \rho(.)
                    gamma_module.weight.data = ori_gnn_weights[i] + gamma_ * ori_gnn_weights[i].relu()
                gnn_gamma_modules.append(gamma_module)

            # --- record original weights of fc layer ---
            ori_fc_weights = []
            fc_gamma_modules = []
            for i, fc_step in enumerate(fc_steps):
                modules = fc_step['module']

                gamma_module = copy.deepcopy(modules[0])
                # Creates modified version of the module

                if hasattr(modules[0], 'weight'):
                    ori_fc_weights.append(modules[0].weight.data)
                    gamma_ = 1 # HARD-CODED - possible hyperparameter
                    # (.)^ = (.) + \gamma * \rho(.)
                    gamma_module.weight.data = ori_fc_weights[i] + gamma_ * ori_fc_weights[i].relu()
                else:
                    ori_fc_weights.append(None) # If layer does not have weight attribute
                fc_gamma_modules.append(gamma_module)

            # --- GNN_LRP implementation ---
            for walk_indices in walk_indices_list: # Iterate over each walk
                # Extract node indices for walk:
                walk_node_indices = [edge_index_with_loop[0, walk_indices[0]]]
                for walk_idx in walk_indices:
                    walk_node_indices.append(edge_index_with_loop[1, walk_idx])

                h = x.requires_grad_(True)
                for i, walk_step in enumerate(walk_steps): # Iterate over layers:
                    modules = walk_step['module']

                    if i == (len(walk_step) - 1): 
                        # Compute h propagation differently if we're at the last GNN layer
                        #print('h shape', h.shape)
                        std_h = GraphSequential(*modules)(h, torch.zeros(2, h.shape[0], dtype=torch.long, device=self.device))

                        s = gnn_gamma_modules[i](h, torch.zeros(2, h.shape[0], dtype=torch.long, device=self.device))
                        ht = (s + epsilon) * (std_h / (s + epsilon)).detach()
                        h = ht

                    else: # Means we're before the last GNN layer

                        if hasattr(modules[0], 'nn'): # Only for GINs:
                            # for the specific 2-layer nn GINs.
                            gin = modules[0]
                            run1 = gin(h, edge_index, probe=True)
                            std_h1 = gin.fc_steps[0]['output']
                            gamma_run1 = gnn_gamma_modules[i](h, edge_index, probe=True)
                            p1 = gnn_gamma_modules[i].fc_steps[0]['output']
                            q1 = (p1 + epsilon) * (std_h1 / (p1 + epsilon)).detach()

                            std_h2 = GraphSequential(*gin.fc_steps[1]['module'])(q1)
                            p2 = GraphSequential(*gnn_gamma_modules[i].fc_steps[1]['module'])(q1)
                            q2 = (p2 + epsilon) * (std_h2 / (p2 + epsilon)).detach()
                            q = q2
                        else: # For GCN layers:

                            std_h = GraphSequential(*modules)(h, edge_index)

                            # --- LRP-gamma ---
                            p = gnn_gamma_modules[i](h, edge_index)
                            q = (p + epsilon) * (std_h / (p + epsilon)).detach()

                        # --- pick a path ---
                        mk = torch.zeros((h.shape[0], 1), device=self.device)
                        k = walk_node_indices[i + 1]
                        mk[k] = 1
                        ht = q * mk + q.detach() * (1 - mk)
                        h = ht

                # --- FC LRP_gamma ---
                for i, fc_step in enumerate(fc_steps): # Compute forward passes over FC for given walk
                    modules = fc_step['module']
                    # std_h = nn.Sequential(*modules)(h) if i != 0 \
                    #     else GraphSequential(*modules)(h, torch.zeros(h.shape[0], dtype=torch.long, device=self.device))
                    std_h = nn.Sequential(*modules)(h)

                    # --- gamma ---
                    # s = fc_gamma_modules[i](h) if i != 0 \
                    #     else fc_gamma_modules[i](h, torch.zeros(h.shape[0], dtype=torch.long, device=self.device))
                    s = fc_gamma_modules[i](h)
                    ht = (s + epsilon) * (std_h / (s + epsilon)).detach()
                    h = ht

                if not self.explain_graph:
                    f = h[node_idx, label]
                else:
                    f = h[0, label]

                # Compute relevance score:
                x_grads = torch.autograd.grad(outputs=f, inputs=x)[0]
                I = walk_node_indices[0]
                r = x_grads[I, :] @ x[I].T
                # Inner-product of gradients of x and x itself:
                # Only use nodes in given walk
                walk_scores.append(r)

        #labels = tuple(i for i in range(kwargs.get('num_classes')))
        labels = tuple(i for i in range(num_classes))
        walk_scores_tensor_list = [None for i in labels]
        for label in labels:

            walk_scores = []

            compute_walk_score()
            walk_scores_tensor_list[label] = torch.stack(walk_scores, dim=0).view(-1, 1)

        walks = {'ids': walk_indices_list, 'score': torch.cat(walk_scores_tensor_list, dim=1)}


        # --- Apply edge mask evaluation ---
        with torch.no_grad():
            with self.connect_mask(self):
                ex_labels = tuple(torch.tensor([label]).to(self.device) for label in labels)
                masks = []
                for ex_label in ex_labels:
                    edge_attr = self.explain_edges_with_loop(x, walks, ex_label)
                    mask = edge_attr
                    mask = self.control_sparsity(mask, kwargs.get('sparsity'))
                    masks.append(mask.detach())

                # if forward_args is None:
                #     related_preds = self.eval_related_pred(x, edge_index, masks, **kwargs)
                # else:
                #     related_preds = self.eval_related_pred(x, edge_index, masks, forward_args=forward_args, **kwargs)

        if get_edge_scores:
            subgraph_edge_mask = khop_info[3]
            mask_inds = subgraph_edge_mask.nonzero(as_tuple=True)[0]
            khop_info = list(khop_info)
            khop_info[1] = edge_index_with_loop[:,mask_inds] # Ensure reordering occurs
            edge_scores = [self.__parse_edges(walks, mask_inds, i, agg = edge_aggregator) for i in labels]

            # edge_scores has same edge score ordering as khop_info[1] (i.e. edge_index of subgraph)
            return edge_scores, khop_info 
        
        return walks, khop_info # Returns scores in terms of walks
        #return walks, masks, related_preds, khop_info
    
    def get_explanation_graph(self,
            x: Tensor,
            edge_index: Tensor,
            num_classes: int,
            forward_args: tuple = None,
            get_edge_scores: bool = True,
            edge_aggregator: Callable[[list], float] = np.sum,
            **kwargs
        ):
        '''
        Get explanation for computational graph around one node.

        .. note:: 
            `edge_aggregator` must take one argument, a list, and return one scalar (float).

        Args:
            x (Tensor): Graph input features.
            num_classes (int): Number of classes in model.
            edge_index (Tensor): Input edge_index of graph.
            forward_args (tuple, optional): Any additional inputs to model.forward method (other 
                than x and edge_index). (default: :obj:`None`)
            get_edge_scores (bool, optional): If true, returns edge scores as combined by
                `edge_aggregator` function. (default: :obj:`True`)
            edge_aggregator (Callable[[list], float], optional): Function to combine scores from 
                multiple walks across one edge. Argument only has effect if `get_edge_scores == True`.
                (default: :obj:`numpy.sum`)

        :rtype: 
            If `get_edge_scores == True`, (:obj:`list`, :obj:`tuple`). Return is list of aggregated
                explanation values for each edge in graph. First dimension of list corresponds to label 
                that those values explain, i.e. `len(exp) == num_classes` with 
                `len(exp[i]) == # nodes in graph` for all i. Second return in tuple is `edge_index_with_loop`,
                the new edge index that is used for computation within the explanation method and model.
            If `get_edge_scores == False`, (:obj:`dict`, :obj:`tuple`). Return is dict of walks, with
                keys `'ids'` and `'scores'`, corresponding to walks as denoted by edge indices and scores
                of those corresponding walks, respectively. Second return is still `edge_index_with_loop`.
        '''

        super().forward(x, edge_index, **kwargs)
        self.model.eval()

        walk_steps, fc_steps = self.extract_step(x, edge_index, detach=False, split_fc=True, 
            forward_args = forward_args)


        edge_index_with_loop, _ = add_self_loops(edge_index, num_nodes=self.num_nodes)


        walk_indices_list = torch.tensor(
            self.walks_pick(edge_index_with_loop.cpu(), list(range(edge_index_with_loop.shape[1])),
                            num_layers=self.num_layers), device=self.device)
        
        def compute_walk_score():

            # hyper-parameter gamma
            epsilon = 1e-30   # prevent from zero division
            gamma = [2, 1, 1] # Hard-coded from DIG
            # TODO: Make this a hyperparameter

            # --- record original weights of GNN ---
            ori_gnn_weights = []
            gnn_gamma_modules = []
            clear_probe = x
            for i, walk_step in enumerate(walk_steps):
                modules = walk_step['module']
                gamma_ = gamma[i] if i <= 1 else 1
                if hasattr(modules[0], 'nn'):
                    clear_probe = modules[0](clear_probe, edge_index, probe=False)
                    # clear nodes that are not created by user

                gamma_module = copy.deepcopy(modules[0])
                # Creates modified version of the module

                if hasattr(modules[0], 'nn'):
                    for j, fc_step in enumerate(gamma_module.fc_steps):
                        fc_modules = fc_step['module']
                        if hasattr(fc_modules[0], 'weight'):
                            ori_fc_weight = fc_modules[0].weight.data
                            fc_modules[0].weight.data = ori_fc_weight + gamma_ * ori_fc_weight
                else:
                    ori_gnn_weights.append(modules[0].weight.data)

                    # (.)^ = (.) + \gamma * \rho(.)
                    gamma_module.weight.data = ori_gnn_weights[i] + gamma_ * ori_gnn_weights[i].relu()
                gnn_gamma_modules.append(gamma_module)

            # --- record original weights of fc layer ---
            ori_fc_weights = []
            fc_gamma_modules = []
            for i, fc_step in enumerate(fc_steps):
                modules = fc_step['module']

                gamma_module = copy.deepcopy(modules[0])
                # Creates modified version of the module

                if hasattr(modules[0], 'weight'):
                    ori_fc_weights.append(modules[0].weight.data)
                    gamma_ = 1 # HARD-CODED - possible hyperparameter
                    # (.)^ = (.) + \gamma * \rho(.)
                    gamma_module.weight.data = ori_fc_weights[i] + gamma_ * ori_fc_weights[i].relu()
                else:
                    ori_fc_weights.append(None) # If layer does not have weight attribute
                fc_gamma_modules.append(gamma_module)

            # --- GNN_LRP implementation ---
            for walk_indices in walk_indices_list: # Iterate over each walk
                # Extract node indices for walk:
                walk_node_indices = [edge_index_with_loop[0, walk_indices[0]]]
                for walk_idx in walk_indices:
                    walk_node_indices.append(edge_index_with_loop[1, walk_idx])

                h = x.requires_grad_(True)
                for i, walk_step in enumerate(walk_steps): # Iterate over layers:
                    modules = walk_step['module']

                    if i == (len(walk_step) - 1): 
                        # Compute h propagation differently if we're at the last GNN layer
                        #print('h shape', h.shape)
                        std_h = GraphSequential(*modules)(h, torch.zeros(2, h.shape[0], dtype=torch.long, device=self.device))

                        s = gnn_gamma_modules[i](h, torch.zeros(2, h.shape[0], dtype=torch.long, device=self.device))
                        ht = (s + epsilon) * (std_h / (s + epsilon)).detach()
                        h = ht

                    else: # Means we're before the last GNN layer

                        if hasattr(modules[0], 'nn'): # Only for GINs:
                            # for the specific 2-layer nn GINs.
                            gin = modules[0]
                            run1 = gin(h, edge_index, probe=True)
                            std_h1 = gin.fc_steps[0]['output']
                            gamma_run1 = gnn_gamma_modules[i](h, edge_index, probe=True)
                            p1 = gnn_gamma_modules[i].fc_steps[0]['output']
                            q1 = (p1 + epsilon) * (std_h1 / (p1 + epsilon)).detach()

                            std_h2 = GraphSequential(*gin.fc_steps[1]['module'])(q1)
                            p2 = GraphSequential(*gnn_gamma_modules[i].fc_steps[1]['module'])(q1)
                            q2 = (p2 + epsilon) * (std_h2 / (p2 + epsilon)).detach()
                            q = q2
                        else: # For GCN layers:

                            std_h = GraphSequential(*modules)(h, edge_index)

                            # --- LRP-gamma ---
                            p = gnn_gamma_modules[i](h, edge_index)
                            q = (p + epsilon) * (std_h / (p + epsilon)).detach()

                        # --- pick a path ---
                        mk = torch.zeros((h.shape[0], 1), device=self.device)
                        k = walk_node_indices[i + 1]
                        mk[k] = 1
                        ht = q * mk + q.detach() * (1 - mk)
                        h = ht

                # --- FC LRP_gamma ---
                for i, fc_step in enumerate(fc_steps): # Compute forward passes over FC for given walk
                    modules = fc_step['module']
                    # std_h = nn.Sequential(*modules)(h) if i != 0 \
                    #     else GraphSequential(*modules)(h, torch.zeros(h.shape[0], dtype=torch.long, device=self.device))
                    std_h = nn.Sequential(*modules)(h)

                    # --- gamma ---
                    # s = fc_gamma_modules[i](h) if i != 0 \
                    #     else fc_gamma_modules[i](h, torch.zeros(h.shape[0], dtype=torch.long, device=self.device))
                    s = fc_gamma_modules[i](h)
                    ht = (s + epsilon) * (std_h / (s + epsilon)).detach()
                    h = ht

                f = h[0, label]

                # Compute relevance score:
                x_grads = torch.autograd.grad(outputs=f, inputs=x)[0]
                I = walk_node_indices[0]
                r = x_grads[I, :] @ x[I].T
                # Inner-product of gradients of x and x itself:
                # Only use nodes in given walk
                walk_scores.append(r)

        #labels = tuple(i for i in range(kwargs.get('num_classes')))
        labels = tuple(i for i in range(num_classes))
        walk_scores_tensor_list = [None for i in labels]
        for label in labels:

            walk_scores = []

            compute_walk_score()
            walk_scores_tensor_list[label] = torch.stack(walk_scores, dim=0).view(-1, 1)

        walks = {'ids': walk_indices_list, 'score': torch.cat(walk_scores_tensor_list, dim=1)}


        # --- Apply edge mask evaluation ---
        with torch.no_grad():
            with self.connect_mask(self):
                ex_labels = tuple(torch.tensor([label]).to(self.device) for label in labels)
                masks = []
                for ex_label in ex_labels:
                    edge_attr = self.explain_edges_with_loop(x, walks, ex_label)
                    mask = edge_attr
                    mask = self.control_sparsity(mask, kwargs.get('sparsity'))
                    masks.append(mask.detach())

                # if forward_args is None:
                #     related_preds = self.eval_related_pred(x, edge_index, masks, **kwargs)
                # else:
                #     related_preds = self.eval_related_pred(x, edge_index, masks, forward_args=forward_args, **kwargs)

        if get_edge_scores:
            edge_ind_range = torch.arange(start = 0, end=edge_index_with_loop.shape[1])
            edge_scores = [self.__parse_edges(walks, edge_ind_range, i, agg = edge_aggregator) for i in labels]
            # Returns edge scores, whose indices correspond to scores for each edge in edge_index_with_loop
            return edge_scores, edge_index_with_loop

        #return walks, masks, related_preds
        return walks, edge_index_with_loop # walks dictionary version of scores


    def forward(self,
                x: Tensor,
                edge_index: Tensor,
                num_classes: int,
                forward_args: tuple = None,
                node_idx: int = None,
                **kwargs
                ):
        r"""
        Run the explainer for a specific graph instance.

        Args:
            x (torch.Tensor): The graph instance's input node features.
            edge_index (torch.Tensor): The graph instance's edge index.
            **kwargs (dict):
                :obj:`node_idx` （int): The index of node that is pending to be explained.
                (for node classification)
                :obj:`sparsity` (float): The Sparsity we need to control to transform a
                soft mask to a hard mask. (Default: :obj:`0.7`)
                :obj:`num_classes` (int): The number of task's classes.

        :rtype:
            (walks, edge_masks, related_predictions),
            walks is a dictionary including walks' edge indices and corresponding explained scores;
            edge_masks is a list of edge-level explanation for each class;
            related_predictions is a list of dictionary for each class
            where each dictionary includes 4 type predicted probabilities.

        """
        super().forward(x, edge_index, **kwargs)
        self.model.eval()

        walk_steps, fc_steps = self.extract_step(x, edge_index, detach=False, split_fc=True, 
            forward_args = forward_args)


        edge_index_with_loop, _ = add_self_loops(edge_index, num_nodes=self.num_nodes)


        walk_indices_list = torch.tensor(
            self.walks_pick(edge_index_with_loop.cpu(), list(range(edge_index_with_loop.shape[1])),
                            num_layers=self.num_layers), device=self.device)
        if not self.explain_graph:
            #node_idx = kwargs.get('node_idx')
            assert node_idx is not None
            khop_info  = k_hop_subgraph(
                node_idx, self.__num_hops__, edge_index_with_loop, relabel_nodes=True,
                num_nodes=None, flow=self.__flow__())

            self.hard_edge_mask = khop_info[-1]

            # walk indices list mask
            edge2node_idx = edge_index_with_loop[1] == node_idx
            walk_indices_list_mask = edge2node_idx[walk_indices_list[:, -1]]
            walk_indices_list = walk_indices_list[walk_indices_list_mask]


        def compute_walk_score():

            # hyper-parameter gamma
            epsilon = 1e-30   # prevent from zero division
            gamma = [2, 1, 1] # Hard-coded from DIG
            # TODO: Make this a hyperparameter

            # --- record original weights of GNN ---
            ori_gnn_weights = []
            gnn_gamma_modules = []
            clear_probe = x
            for i, walk_step in enumerate(walk_steps):
                modules = walk_step['module']
                gamma_ = gamma[i] if i <= 1 else 1
                if hasattr(modules[0], 'nn'):
                    clear_probe = modules[0](clear_probe, edge_index, probe=False)
                    # clear nodes that are not created by user

                gamma_module = copy.deepcopy(modules[0])
                # Creates modified version of the module

                if hasattr(modules[0], 'nn'):
                    for j, fc_step in enumerate(gamma_module.fc_steps):
                        fc_modules = fc_step['module']
                        if hasattr(fc_modules[0], 'weight'):
                            ori_fc_weight = fc_modules[0].weight.data
                            fc_modules[0].weight.data = ori_fc_weight + gamma_ * ori_fc_weight
                else:
                    ori_gnn_weights.append(modules[0].weight.data)

                    # (.)^ = (.) + \gamma * \rho(.)
                    gamma_module.weight.data = ori_gnn_weights[i] + gamma_ * ori_gnn_weights[i].relu()
                gnn_gamma_modules.append(gamma_module)

            # --- record original weights of fc layer ---
            ori_fc_weights = []
            fc_gamma_modules = []
            for i, fc_step in enumerate(fc_steps):
                modules = fc_step['module']

                gamma_module = copy.deepcopy(modules[0])
                # Creates modified version of the module

                if hasattr(modules[0], 'weight'):
                    ori_fc_weights.append(modules[0].weight.data)
                    gamma_ = 1 # HARD-CODED - possible hyperparameter
                    # (.)^ = (.) + \gamma * \rho(.)
                    gamma_module.weight.data = ori_fc_weights[i] + gamma_ * ori_fc_weights[i].relu()
                else:
                    ori_fc_weights.append(None) # If layer does not have weight attribute
                fc_gamma_modules.append(gamma_module)

            # --- GNN_LRP implementation ---
            for walk_indices in walk_indices_list: # Iterate over each walk
                # Extract node indices for walk:
                walk_node_indices = [edge_index_with_loop[0, walk_indices[0]]]
                for walk_idx in walk_indices:
                    walk_node_indices.append(edge_index_with_loop[1, walk_idx])

                h = x.requires_grad_(True)
                for i, walk_step in enumerate(walk_steps): # Iterate over layers:
                    modules = walk_step['module']

                    if i == (len(walk_step) - 1): 
                        # Compute h propagation differently if we're at the last GNN layer
                        #print('h shape', h.shape)
                        std_h = GraphSequential(*modules)(h, torch.zeros(2, h.shape[0], dtype=torch.long, device=self.device))

                        s = gnn_gamma_modules[i](h, torch.zeros(2, h.shape[0], dtype=torch.long, device=self.device))
                        ht = (s + epsilon) * (std_h / (s + epsilon)).detach()
                        h = ht

                    else: # Means we're before the last GNN layer

                        if hasattr(modules[0], 'nn'): # Only for GINs:
                            # for the specific 2-layer nn GINs.
                            gin = modules[0]
                            run1 = gin(h, edge_index, probe=True)
                            std_h1 = gin.fc_steps[0]['output']
                            gamma_run1 = gnn_gamma_modules[i](h, edge_index, probe=True)
                            p1 = gnn_gamma_modules[i].fc_steps[0]['output']
                            q1 = (p1 + epsilon) * (std_h1 / (p1 + epsilon)).detach()

                            std_h2 = GraphSequential(*gin.fc_steps[1]['module'])(q1)
                            p2 = GraphSequential(*gnn_gamma_modules[i].fc_steps[1]['module'])(q1)
                            q2 = (p2 + epsilon) * (std_h2 / (p2 + epsilon)).detach()
                            q = q2
                        else: # For GCN layers:

                            std_h = GraphSequential(*modules)(h, edge_index)

                            # --- LRP-gamma ---
                            p = gnn_gamma_modules[i](h, edge_index)
                            q = (p + epsilon) * (std_h / (p + epsilon)).detach()

                        # --- pick a path ---
                        mk = torch.zeros((h.shape[0], 1), device=self.device)
                        k = walk_node_indices[i + 1]
                        mk[k] = 1
                        ht = q * mk + q.detach() * (1 - mk)
                        h = ht

                # --- FC LRP_gamma ---
                for i, fc_step in enumerate(fc_steps): # Compute forward passes over FC for given walk
                    modules = fc_step['module']
                    # std_h = nn.Sequential(*modules)(h) if i != 0 \
                    #     else GraphSequential(*modules)(h, torch.zeros(h.shape[0], dtype=torch.long, device=self.device))
                    std_h = nn.Sequential(*modules)(h)

                    # --- gamma ---
                    # s = fc_gamma_modules[i](h) if i != 0 \
                    #     else fc_gamma_modules[i](h, torch.zeros(h.shape[0], dtype=torch.long, device=self.device))
                    s = fc_gamma_modules[i](h)
                    ht = (s + epsilon) * (std_h / (s + epsilon)).detach()
                    h = ht

                if not self.explain_graph:
                    f = h[node_idx, label]
                else:
                    f = h[0, label]

                # Compute relevance score:
                x_grads = torch.autograd.grad(outputs=f, inputs=x)[0]
                I = walk_node_indices[0]
                r = x_grads[I, :] @ x[I].T
                # Inner-product of gradients of x and x itself:
                # Only use nodes in given walk
                walk_scores.append(r)

        #labels = tuple(i for i in range(kwargs.get('num_classes')))
        labels = tuple(i for i in range(num_classes))
        walk_scores_tensor_list = [None for i in labels]
        for label in labels: # Compute label-wise scores:

            walk_scores = []

            compute_walk_score()
            walk_scores_tensor_list[label] = torch.stack(walk_scores, dim=0).view(-1, 1)

        walks = {'ids': walk_indices_list, 'score': torch.cat(walk_scores_tensor_list, dim=1)}


        # --- Apply edge mask evaluation ---
        with torch.no_grad():
            with self.connect_mask(self):
                ex_labels = tuple(torch.tensor([label]).to(self.device) for label in labels)
                masks = []
                for ex_label in ex_labels:
                    edge_attr = self.explain_edges_with_loop(x, walks, ex_label)
                    mask = edge_attr
                    mask = self.control_sparsity(mask, kwargs.get('sparsity'))
                    masks.append(mask.detach())

                if forward_args is None:
                    related_preds = self.eval_related_pred(x, edge_index, masks, **kwargs)
                else:
                    related_preds = self.eval_related_pred(x, edge_index, masks, forward_args=forward_args, **kwargs)

        if not self.explain_graph:
            #subgraph_nodes = khop_info[0]
            #subgraph_eidx = khop_info[1]
            subgraph_edge_mask = khop_info[3]
            mask_inds = subgraph_edge_mask.nonzero(as_tuple=True)[0]
            khop_info = list(khop_info)
            khop_info[1] = edge_index_with_loop[:,mask_inds] # Ensure reordering occurs
            edge_scores = [self.__parse_edges(walks, mask_inds, i) for i in labels]

            #return walks, masks, related_preds, khop_info
            return edge_scores, khop_info 
        else:
            return walks, masks, related_preds

    def __parse_edges(self, walks: dict, mask_inds: torch.Tensor, label_idx: int, 
        agg: Callable[[list], float] = np.sum):
        '''
        Retrieves and aggregates all walk scores into concise edge scores.
        Args:
            walks (dict): Contains 'ids' and 'score' keys as returned by functions.
            mask_inds (torch.Tensor): Edges to compute walk score over
            label_idx (int): index of label
            agg (Callable[[list], float]): Aggregation function for edge scores
        '''
        walk_ids = walks['ids']
        walk_scores = walks['score']

        #unique_edges = walks.unique().tolist()
        #edge_maps = {i.item():[] for i in mask_inds}
        edge_maps = [[] for _ in mask_inds]

        for i in range(walk_ids.shape[0]): # Over all walks:
            walk = walk_ids[i,:]
            score = walk_scores[i,label_idx].item()
            #walk_edges = walk.unique().tolist()
            for wn in walk:
                index_in_mask = (mask_inds == wn.item()).nonzero(as_tuple=True)[0].item()
                edge_maps[index_in_mask].append(score)

        edge_scores = [agg(e) for e in edge_maps]
        return edge_scores

    def __parse_GNNLRP_explanations(self, walks, edge_index, label_idx, 
        edge_agg = np.sum, node_agg = lambda x: np.sum(x)):

        #walks, edge_masks, related_predictions = ret_tuple # Unpack

        walk_ids = walks['ids']
        walk_scores = walks['score']

        unique_edges = walk_ids.unique().tolist()
        edge_maps = [[] if e in unique_edges else [0] for e in range(edge_masks[0].shape[0])]
        #edge_maps = list(np.zeros(edge_masks[0].shape[0]))
        edge_maps = [[] for w in walk_ids.unique().tolist()]
        #edge_maps = {w:[] for w in walk_ids.unique().tolist()}
        # Use list to preserve order

        for i in range(walk_ids.shape[0]): # Over all walks:
            walk = walk_ids[i,:]
            score = walk_scores[i,label_idx].item()
            walk_nodes = walk.unique().tolist()
            for wn in walk_nodes:
                edge_maps[wn].append(score)

        N = maybe_num_nodes(edge_index)
        node_map = [[] for i in range(N)]

        # Aggregate all edge scores:
        # edge_scores = np.zeros(edge_masks[0].shape[0])
        # for w, x in edge_maps.items():
        #     edge_scores[w] = edge_agg(x)
        edge_scores = [edge_agg(x) for x in edge_maps]
        #edge_scores = list(edge_scores)

        # Iterate over edges, put into their respective portions:
        # Combines edges in given form:
        for n in range(N):
            edge_inds = all_incoming_edges_w_node(edge_index, n)
            for e in edge_inds:
                if isinstance(edge_maps[e], Iterable):
                    node_map[n].append(edge_scores[e]) # Append edge scores
                else:
                    node_map[n].append([edge_scores[e]])

        # Now combine all incoming edge scores for nodes:
        node_scores = [sum([abs(xi) for xi in x]) for x in node_map]
        print('len node_scores', len(node_scores))

        return node_scores, edge_scores

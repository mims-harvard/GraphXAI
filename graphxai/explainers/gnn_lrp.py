import torch
from torch import Tensor
import torch.nn as nn
import copy
from typing import Callable, Tuple
from torch_geometric.nn import GCNConv
from torch_geometric.utils.loop import add_self_loops
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data

from ._decomp_base import _BaseDecomposition
from graphxai.utils import Explanation

device = "cuda" if torch.cuda.is_available() else "cpu"

EPS = 1e-15

def all_incoming_edges_w_node(edge_index, node_idx, row = 0):
    '''Gets all incoming edges to a given node provided row index'''
    return (edge_index[row,:] == node_idx).nonzero(as_tuple=True)[0].tolist()

class GraphSequential(nn.Sequential):

    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, *input) -> Tensor:
        for module in self:
            #print(input)
            #print(module)
            if isinstance(input, tuple):
                print('input', input)
                print(type(module))
                input = module(*input)
            else:
                input = module(input)
            #print(input)
            #print(module)
        return input

#class GNN_LRP(WalkBase):
class GNN_LRP(_BaseDecomposition):
    r"""
    Code adapted from Dive into Graphs (DIG)
    Code: https://github.com/divelab/DIG

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

    def __init__(self, model: nn.Module):
        super().__init__(model=model)

    def get_explanation_node(self, 
            x: Tensor, 
            edge_index: Tensor,
            node_idx: int,
            label: int = None,
            forward_kwargs: dict = {},
            get_walk_scores: bool = False,
            edge_aggregator: Callable[[torch.Tensor], torch.Tensor] = torch.mean,
            **kwargs
        ) -> Tuple[dict, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
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
            forward_kwargs (dict, optional): Additional arguments to model.forward 
                beyond x and edge_index. Must be keyed on argument name. 
                (default: :obj:`{}`)
            get_walk_scores (bool, optional): If true, returns scores for each individual walk. 
                (default: :obj:`False`)
            edge_aggregator (Callable[[torch.Tensor], torch.Tensor], optional): Function to combine scores from 
                multiple walks across one edge. Must take a shape (n,) tensor and output a tensor of a single element 
                (i.e. dimension 0 tensor). Other examples include `torch.sum` and `torch.norm`. Argument only has 
                effect if `get_walk_scores == False`.
                (default: :obj:`torch.mean`)

        :rtype: (:obj:`dict`, (:class:`torch.Tensor`, :class:`torch.Tensor`, :class:`torch.Tensor`, :class:`torch.Tensor`))

        Returns:
            exp (dict):
                exp['feature_imp'] is :obj:`None` because no feature explanations are generated.
                exp['node_imp'] is :obj:`None` because no node explanations are generated.
                If `get_edge_scores == True`: 
                    exp['edge_imp'] (torch.Tensor): tensor of shape `(e,)` where `e` is the number of edges 
                        in the k-hop subgraph of node specified by `node_idx`. These are each aggregated scores 
                        from the combinations of walks over each edge.
                If `get_edge_scores == False`:
                    exp['edge_imp'] (dict): dict containing keys `ids` and `scores`, where `ids` are indices of edges in the original
                        `edge_index` with added self-loops; these values correspond to walks on the graph. The `scores`
                        key consists of a `(1,w)` tensor, where `w` is the number of walks on the k-hop subgraph around
                        node_idx; these values are each scores for each of their corresponding walks in the `ids` key.
            khop_info (4-tuple of torch.Tensor):
                0. the nodes involved in the subgraph
                1. the filtered `edge_index`
                2. the mapping from node indices in `node_idx` to their new location
                3. the `edge_index` mask indicating which edges were preserved 
        '''

        #super().forward(x, edge_index, **kwargs)
        super().set_graph_attr(x, edge_index, explain_graph = False, **kwargs)
        #self.set_graph_attr(x, edge_index, explain_graph = False, **kwargs)
        self.model.eval()

        # Ensure types:
        label = int(self.model(x, edge_index, **forward_kwargs).argmax(dim=1).item()) if label is None else int(label)
        node_idx = int(node_idx)

        # Step through the model:
        walk_steps, fc_steps = self.extract_step(x, edge_index, detach=False, split_fc=True, 
            forward_kwargs = forward_kwargs)

        edge_index_with_loop, _ = add_self_loops(edge_index, num_nodes=self.num_nodes)

        print(self.device)
        walk_indices_list = torch.tensor(
            self.walks_pick(edge_index_with_loop, list(range(edge_index_with_loop.shape[1])),
                            num_layers=self.L), device=self.device)

        # Get subgraph of nodes in computational graph:
        khop_info  = k_hop_subgraph(
            node_idx, self.L, edge_index_with_loop, relabel_nodes=True,
            num_nodes=None, flow=self._flow())

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
            print('walkstep', [walk_steps[i]['module'] for i in range(len(walk_steps))])
            print('fcstep', [fc_steps[i]['module'] for i in range(len(fc_steps))])
            for i, walk_step in enumerate(walk_steps):
                modules = walk_step['module']
                gamma_ = gamma[i] if i <= 1 else 1
                if hasattr(modules[0], 'nn'):
                    clear_probe = modules[0](clear_probe, edge_index) #probe=False)
                    # clear nodes that are not created by user

                gamma_module = copy.deepcopy(modules[0])
                # Creates modified version of the module

                if hasattr(modules[0], 'nn'):
                    for j, fc_step in enumerate(gamma_module.fc_steps):
                        fc_modules = fc_step['module']
                        if hasattr(fc_modules[0], 'weight'):
                            ori_fc_weight = fc_modules[0].weight.data
                            fc_modules[0].weight.data = ori_fc_weight + gamma_ * ori_fc_weight
                
                elif isinstance(modules[0], GCNConv):
                    ori_gnn_weights.append(modules[0].lin.weight.data)
                    # (.)^ = (.) + \gamma * \rho(.)
                    gamma_module.lin.weight.data = ori_gnn_weights[i] + gamma_ * ori_gnn_weights[i].relu()
                else: # Should be Linear layer
                    ori_gnn_weights.append(modules[0].weight.data)
                    # (.)^ = (.) + \gamma * \rho(.)
                    gamma_module.weight.data = ori_gnn_weights[i] + gamma_ * ori_gnn_weights[i].relu()

                    
                    #gamma_module.weight.data = ori_gnn_weights[i] + gamma_ * ori_gnn_weights[i].relu()
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
                    print('walk step modules', modules)

                    # if i == (len(walk_step) - 1): 
                    #     # Compute h propagation differently if we're at the last GNN layer
                    #     std_h = GraphSequential(*modules)(h, torch.zeros(2, h.shape[0], dtype=torch.long, device=self.device))

                    #     s = gnn_gamma_modules[i](h, torch.zeros(2, h.shape[0], dtype=torch.long, device=self.device))
                    #     ht = (s + epsilon) * (std_h / (s + epsilon)).detach()
                    #     h = ht

                    #else: # Means we're before the last GNN layer

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
                        print('Layer {}'.format(i))
                        print('size p', p.shape)
                        print('size std_h', std_h.shape)
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
                    # **************************
                    std_h = nn.Sequential(*modules)(h) if i != 0 \
                            else GraphSequential(*modules)(h, torch.zeros(h.shape[0], dtype=torch.long, device=self.device))

                    # --- gamma ---
                    s = fc_gamma_modules[i](h) if i != 0 \
                            else fc_gamma_modules[i](h, torch.zeros(h.shape[0], dtype=torch.long, device=self.device))
                    ht = (s + epsilon) * (std_h / (s + epsilon)).detach()
                    h = ht

                # if not self.explain_graph:
                #     f = h[node_idx, label]
                
                f = h[0, label]

                # Compute relevance score:
                x_grads = torch.autograd.grad(outputs=f, inputs=x)[0]
                I = walk_node_indices[0]
                r = x_grads[I, :] @ x[I].T
                # Inner-product of gradients of x and x itself:
                # Only use nodes in given walk
                walk_scores.append(r)

        walk_scores = []

        compute_walk_score()
        walk_scores_tensor_list = torch.stack(walk_scores, dim=0).view(-1, 1)

        walks = {'ids': walk_indices_list, 'score': walk_scores_tensor_list}

        # if get_walk_scores:
        #     return {'feature_imp': None, 'node_imp': None, 'edge_imp': walks}, khop_info # Returns scores in terms of walks
        
        # if returning edge scores:
        subgraph_edge_mask = khop_info[3]
        mask_inds = subgraph_edge_mask.nonzero(as_tuple=True)[0]
        khop_info = list(khop_info)
        khop_info[1] = edge_index_with_loop[:,mask_inds] # Ensure reordering occurs
        edge_scores = self.__parse_edges(walks, mask_inds, label, agg = edge_aggregator)

        exp = Explanation(
            edge_imp = edge_scores,
            node_idx = node_idx
        )
        # exp.edge_imp = edge_scores
        # exp.node_idx = node_idx
        exp.set_enclosing_subgraph(khop_info)
        #exp.set_whole_graph(x, edge_index_with_loop)

        # Method-specific attributes:
        exp._walk_ids = walks['ids']
        exp._walk_scores = walks['score']

        # edge_scores has same edge score ordering as khop_info[1] (i.e. edge_index of subgraph)
        #return {'feature_imp': None, 'node_imp': None, 'edge_imp': edge_scores}, khop_info 
        return exp
        
    
    def get_explanation_graph(self,
            x: Tensor,
            edge_index: Tensor,
            label: int = None,
            forward_kwargs: dict = {},
            get_walk_scores: bool = False,
            edge_aggregator: Callable[[torch.Tensor], torch.Tensor] = torch.mean,
            **kwargs
        ):
        '''
        Get explanation for computational graph around one node.

        .. note:: 
            `edge_aggregator` must take one argument, a list, and return one scalar (float).

        Args:
            x (Tensor): Graph input features.
            edge_index (Tensor): Input edge_index of graph.
            forward_kwargs (dict, optional): Additional arguments to model.forward 
                beyond x and edge_index. Must be keyed on argument name. 
                (default: :obj:`{}`)
            get_walk_scores (bool, optional): If true, returns scores for each individual walk. 
                (default: :obj:`False`)
            edge_aggregator (Callable[[torch.Tensor], torch.Tensor], optional): Function to combine scores from 
                multiple walks across one edge. Must take a shape (n,) tensor and output a tensor of a single element 
                (i.e. dimension 0 tensor). Other examples include `torch.sum` and `torch.norm`. Argument only has 
                effect if `get_walk_scores == False`.
                (default: :obj:`torch.mean`)

        :rtype: :class:`dict`
        
        Returns:
            exp (dict):
                exp['feature_imp'] is :obj:`None` because no feature explanations are generated.
                exp['node_imp'] is :obj:`None` because no node explanations are generated.
                If `get_edge_scores == True`: 
                    exp['edge_imp'] (torch.Tensor): tensor of shape `(e,)` where `e` is the number of edges in the graph. 
                        These are each aggregated scores from the combinations of walks over each edge.
                If `get_edge_scores == False`:
                    exp['edge_imp'] (dict): dict containing keys `ids` and `scores`, where `ids` are indices of edges in the original
                        `edge_index` with added self-loops; these values correspond to walks on the graph. The `scores`
                        key consists of a `(1,w)` tensor, where `w` is the number of walks on the graph; these values are each 
                        scores for each of their corresponding walks in the `ids` key.
        '''

        #super().forward(x, edge_index, **kwargs)
        super().set_graph_attr(x, edge_index, explain_graph = True, **kwargs)
        self.model.eval()

        # Ensure label is an int
        label = int(self.model(x, edge_index, **forward_kwargs).argmax(dim=1).item()) if label is None else int(label)

        walk_steps, fc_steps = self.extract_step(x, edge_index, detach=False, split_fc=True, 
            forward_kwargs = forward_kwargs)

        edge_index_with_loop, _ = add_self_loops(edge_index, num_nodes=self.num_nodes)

        walk_indices_list = torch.tensor(
            self.walks_pick(edge_index_with_loop, list(range(edge_index_with_loop.shape[1])),
                            num_layers=self.L), device=self.device)
        
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
                    std_h = nn.Sequential(*modules)(h)

                    # --- gamma ---
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

        walk_scores = []
        compute_walk_score()
        walk_scores_tensor_list = torch.stack(walk_scores, dim=0).view(-1, 1)

        walks = {'ids': walk_indices_list, 'score': walk_scores_tensor_list} # One scalar score for each walk

        # if get_walk_scores:
        #     return {'feature_imp': None, 'node_imp':None, 'edge_imp': walks}, edge_index_with_loop # walks dictionary version of scores

        # If returning edge scores:
        edge_ind_range = torch.arange(start = 0, end=edge_index_with_loop.shape[1])
        edge_scores = self.__parse_edges(walks, edge_ind_range, label, agg = edge_aggregator)

        exp = Explanation(
            edge_imp = edge_scores
        )
        #exp.edge_imp = edge_scores
        exp.set_whole_graph(Data(x=x, edge_index=edge_index_with_loop)) # Make graph with self-loops
        exp._walk_ids = walks['ids']
        exp._walk_scores = walks['score']

        # Returns edge scores, whose indices correspond to scores for each edge in edge_index_with_loop
        #return {'feature_imp': None, 'node_imp': None, 'edge_imp': edge_scores}, edge_index_with_loop
        return exp

    def __parse_edges(self, walks: dict, mask_inds: torch.Tensor, label_idx: int, 
        agg: Callable[[list], float] = torch.mean):
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

        edge_maps = [[] for _ in mask_inds]

        for i in range(walk_ids.shape[0]): # Over all walks:
            walk = walk_ids[i,:]
            if walk_scores.shape[1] > 1:
                score = walk_scores[i,label_idx].item()
            else:
                score = walk_scores[i].item()
            for wn in walk:
                index_in_mask = (mask_inds == wn.item()).nonzero(as_tuple=True)[0].item()
                edge_maps[index_in_mask].append(score)

        edge_scores = [agg(torch.tensor(e)).item() for e in edge_maps]
        return torch.tensor(edge_scores)

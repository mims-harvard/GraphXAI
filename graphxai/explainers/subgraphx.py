import torch
from torch import Tensor
from functools import partial
import torch.nn.functional as F
from typing import Callable, Optional, Tuple
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data

#from .subgraphx_utils.shapley import GnnNets_GC2value_func, GnnNets_NC2value_func
from .subgraphx_utils.subgraphx_fns import find_closest_node_result, reward_func, MCTS

from ._base import _BaseExplainer
from graphxai.utils import Explanation

device = "cuda" if torch.cuda.is_available() else "cpu"

class SubgraphX(_BaseExplainer):
    r"""
    Code adapted from Dive into Graphs (DIG)
    Code: https://github.com/divelab/DIG

    The implementation of paper
    `On Explainability of Graph Neural Networks via Subgraph Explorations <https://arxiv.org/abs/2102.05152>`_.

    Args:
        model (:obj:`torch.nn.Module`): The target model prepared to explain
        num_hops(:obj:`int`, :obj:`None`): The number of hops to extract neighborhood of target node
          (default: :obj:`None`)
        rollout(:obj:`int`): Number of iteration to get the prediction
        min_atoms(:obj:`int`): Number of atoms of the leaf node in search tree
        c_puct(:obj:`float`): The hyperparameter which encourages the exploration
        expand_atoms(:obj:`int`): The number of atoms to expand
          when extend the child nodes in the search tree
        high2low(:obj:`bool`): Whether to expand children nodes from high degree to low degree when
          extend the child nodes in the search tree (default: :obj:`False`)
        local_radius(:obj:`int`): Number of local radius to calculate :obj:`l_shapley`, :obj:`mc_l_shapley`
        sample_num(:obj:`int`): Sampling time of monte carlo sampling approximation for
          :obj:`mc_shapley`, :obj:`mc_l_shapley` (default: :obj:`mc_l_shapley`)
        reward_method(:obj:`str`): The command string to select the
        subgraph_building_method(:obj:`str`): The command string for different subgraph building method,
          such as :obj:`zero_filling`, :obj:`split` (default: :obj:`zero_filling`)

    Example:
        >>> # For graph classification task
        >>> subgraphx = SubgraphX(model=model, num_classes=2)
        >>> _, explanation_results, related_preds = subgraphx(x, edge_index)

    """
    def __init__(self, model, num_hops: Optional[int] = None,
                 rollout: int = 10, min_atoms: int = 3, c_puct: float = 10.0, expand_atoms=14,
                 high2low=False, local_radius=4, sample_num=100, reward_method='mc_l_shapley',
                 subgraph_building_method='zero_filling'):

        super().__init__(model=model, is_subgraphx=True)
        self.model.eval()
        self.num_hops = self.update_num_hops(num_hops)

        # mcts hyper-parameters
        self.rollout = rollout
        self.min_atoms = min_atoms # N_{min}
        self.c_puct = c_puct
        self.expand_atoms = expand_atoms
        self.high2low = high2low

        # reward function hyper-parameters
        self.local_radius = local_radius
        self.sample_num = sample_num
        self.reward_method = reward_method
        self.subgraph_building_method = subgraph_building_method

    def update_num_hops(self, num_hops):
        if num_hops is not None:
            return num_hops

        k = 0
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                k += 1
        return k

    def get_reward_func(self, value_func, node_idx=None, explain_graph = False):
        if explain_graph:
            node_idx = None
        else:
            assert node_idx is not None
        return reward_func(reward_method=self.reward_method,
                           value_func=value_func,
                           node_idx=node_idx,
                           local_radius=self.local_radius,
                           sample_num=self.sample_num,
                           subgraph_building_method=self.subgraph_building_method)

    def get_mcts_class(self, x, edge_index, node_idx: int = None, score_func: Callable = None, explain_graph = False):
        if explain_graph:
            node_idx = None
        else:
            assert node_idx is not None
        return MCTS(x, edge_index,
                    node_idx=node_idx,
                    score_func=score_func,
                    num_hops=self.num_hops,
                    n_rollout=self.rollout,
                    min_atoms=self.min_atoms,
                    c_puct=self.c_puct,
                    expand_atoms=self.expand_atoms,
                    high2low=self.high2low)

    def get_explanation_node(self, 
            x: Tensor, 
            edge_index: Tensor, 
            node_idx: int, 
            label: int = None, 
            y = None,
            max_nodes: int = 14, 
            forward_kwargs: dict = {}
        ) -> Tuple[dict, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        '''
        Get explanation for a single node within a graph.
        Args:
            x (torch.Tensor): Input features for every node in the graph.
            edge_index (torch.Tensor): Edge index for entire input graph.
            node_idx (int): Node index for which to generate an explanation.
            label (int, optional): Label for which to assume as a prediction from 
                the model when generating an explanation. If `None`, this argument 
                is set to the prediction directly from the model. (default: :obj:`None`)
            max_nodes (int, optional): Maximum number of nodes to include in the subgraph 
                generated from the explanation. (default: :obj:`14`)
            forward_kwargs (dict, optional): Additional arguments to model.forward 
                beyond x and edge_index. Must be keyed on argument name. 
                (default: :obj:`{}`)

        :rtype: :class:`Explanation`
        Returns:
            exp (dict):
                exp['feature_imp'] is `None` because no feature explanations are generated.
                exp['node_imp'] (torch.Tensor, (n,)): Node mask of size `(n,)` where `n` 
                    is number of nodes in the entire graph described by `edge_index`. 
                    Type is `torch.bool`, with `True` indices corresponding to nodes 
                    included in the subgraph.
                exp['edge_imp'] (torch.Tensor, (e,)): Edge mask of size `(e,)` where `e` 
                    is number of edges in the entire graph described by `edge_index`. 
                    Type is `torch.bool`, with `True` indices corresponding to edges 
                    included in the subgraph.
            khop_info (4-tuple of torch.Tensor):
                0. the nodes involved in the subgraph
                1. the filtered `edge_index`
                2. the mapping from node indices in `node_idx` to their new location
                3. the `edge_index` mask indicating which edges were preserved 
        '''

        if y is not None:
            label = y[node_idx] # Get node_idx of label

        if label is None:
            self.model.eval()
            pred = self.model(x.to(device), edge_index.to(device), **forward_kwargs)
            label = int(pred.argmax(dim=1).item())
        else:
            label = int(label)

        # collect all the class index
        logits = self.model(x.to(device), edge_index.to(device), **forward_kwargs)
        probs = F.softmax(logits, dim=-1)
        probs = probs.squeeze()

        prediction = probs[node_idx].argmax(-1)
        self.mcts_state_map = self.get_mcts_class(x, edge_index, node_idx=node_idx)
        self.node_idx = self.mcts_state_map.node_idx
        # mcts will extract the subgraph and relabel the nodes
        # value_func = GnnNets_NC2value_func(self.model,
        #                                     node_idx=self.mcts_state_map.node_idx,
        #                                     target_class=label)
        value_func = self._prob_score_func_node(
            node_idx = self.mcts_state_map.node_idx,
            target_class = label
        )
        #value_func = partial(value_func, forward_kwargs=forward_kwargs)
        def wrap_value_func(data):
            return value_func(x=data.x.to(device), edge_index=data.edge_index.to(device), forward_kwargs=forward_kwargs)

        payoff_func = self.get_reward_func(wrap_value_func, node_idx=self.mcts_state_map.node_idx, explain_graph = False)
        self.mcts_state_map.set_score_func(payoff_func)
        results = self.mcts_state_map.mcts(verbose=False)

        # Get best result that has less than max nodes:
        best_result = find_closest_node_result(results, max_nodes=max_nodes)

        # Need to parse results:
        node_mask, edge_mask = self.__parse_results(best_result, edge_index)

        print('args', node_idx, self.L, edge_index)
        khop_info = k_hop_subgraph(node_idx, self.L, edge_index)
        subgraph_edge_mask = khop_info[3] # Mask over edges

        # Set explanation
        exp = Explanation(
            node_imp = 1*node_mask[khop_info[0]], # Apply node mask
            edge_imp = 1*edge_mask[subgraph_edge_mask],
            node_idx = node_idx
        )

        exp.set_enclosing_subgraph(khop_info)

        return exp

    def get_explanation_graph(self, 
            x: Tensor, 
            edge_index: Tensor, 
            label: int = None, 
            max_nodes: int = 14, 
            forward_kwargs: dict = {}, 
        ):
        '''
        Get explanation for a whole graph prediction.
        Args:
            x (torch.Tensor): Input features for every node in the graph.
            edge_index (torch.Tensor): Edge index for entire input graph.
            label (int, optional): Label for which to assume as a prediction from 
                the model when generating an explanation. If `None`, this argument 
                is set to the prediction directly from the model. (default: :obj:`None`)
            max_nodes (int, optional): Maximum number of nodes to include in the subgraph 
                generated from the explanation. (default: :obj:`14`)
            forward_kwargs (dict, optional): Additional arguments to model.forward 
                beyond x and edge_index. Must be keyed on argument name. 
                (default: :obj:`{}`)

        :rtype: :class:`Explanation`
        Returns:
            exp (dict):
                exp['feature_imp'] is `None` because no feature explanations are generated.
                exp['node_imp'] (torch.Tensor, (n,)): Node mask of size `(n,)` where `n` 
                    is number of nodes in the entire graph described by `edge_index`. 
                    Type is `torch.bool`, with `True` indices corresponding to nodes 
                    included in the subgraph.
                exp['edge_imp'] (torch.Tensor, (e,)): Edge mask of size `(e,)` where `e` 
                    is number of edges in the entire graph described by `edge_index`. 
                    Type is `torch.bool`, with `True` indices corresponding to edges 
                    included in the subgraph.
        '''
        if label is None:
            self.model.eval()
            pred = self.model(x, edge_index, **forward_kwargs)
            label = int(pred.argmax(dim=1).item())

        # collect all the class index
        logits = self.model(x, edge_index, **forward_kwargs)
        probs = F.softmax(logits, dim=-1)
        probs = probs.squeeze()

        prediction = probs.argmax(-1)
        
        value_func = self._prob_score_func_graph(target_class = label)
        def wrap_value_func(data):
            return value_func(x=data.x, edge_index=data.edge_index, forward_kwargs=forward_kwargs)

        payoff_func = self.get_reward_func(wrap_value_func, explain_graph = True)
        self.mcts_state_map = self.get_mcts_class(x, edge_index, score_func=payoff_func, explain_graph = True)
        results = self.mcts_state_map.mcts(verbose=False)

        best_result = find_closest_node_result(results, max_nodes=max_nodes)

        node_mask, edge_mask = self.__parse_results(best_result, edge_index)

        exp = Explanation(
            node_imp = node_mask.float(),
            edge_imp = edge_mask.float()
        )
        # exp.node_imp = node_mask
        # exp.edge_imp = edge_mask
        exp.set_whole_graph(Data(x=x, edge_index=edge_index))

        #return {'feature_imp': None, 'node_imp': node_mask, 'edge_imp': edge_mask}
        return exp

    def __parse_results(self, best_subgraph, edge_index):
        # Function strongly based on torch_geometric.utils.subgraph function
        # Citation: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/subgraph.html#subgraph

        # Get mapping
        map = best_subgraph.mapping

        all_nodes = torch.unique(edge_index)

        subgraph_nodes = torch.tensor([map[c] for c in best_subgraph.coalition], dtype=torch.long) if map is not None \
            else torch.tensor(best_subgraph.coalition, dtype=torch.long)

        # Create node mask:
        node_mask = torch.zeros(all_nodes.shape, dtype=torch.bool)
        node_mask[subgraph_nodes] = 1

        # Create edge_index mask
        num_nodes = maybe_num_nodes(edge_index)
        n_mask = torch.zeros(num_nodes, dtype = torch.bool)
        n_mask[subgraph_nodes] = 1

        edge_mask = n_mask[edge_index[0]] & n_mask[edge_index[1]]
        return node_mask, edge_mask

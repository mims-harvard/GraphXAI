import torch
import torch.nn.functional as F
from typing import Tuple
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import k_hop_subgraph

from ._explanation import Explanation
from ._decomp_base import _BaseDecomposition

def clip_hook(grad):
    # Apply ReLU activation to gradient
    return torch.clamp(grad, min=0)#F.relu(grad)

def matching_explanations(nodes, exp):
    # Get explanation matching to subgraph nodes
    new_exp = torch.zeros(len(exp))
    list_nodes = nodes.tolist()
    for i in range(len(exp)):
        if i in nodes:
            new_exp[i] = exp[i]

    return new_exp.tolist()

class GuidedBP(_BaseDecomposition):

    def __init__(self, model, criterion = F.cross_entropy):
        '''
        Args:
            model (torch.nn.Module): model on which to make predictions
            criterion (PyTorch Loss Function): loss function used to train the model.
                Needed to pass gradients backwards in the network to obtain gradients.
        '''
        super().__init__(model)
        self.model = model
        self.criterion = criterion

        self.L = len([module for module in self.model.modules() if isinstance(module, MessagePassing)])

        self.registered_hooks = []

    def get_explanation_node(self, 
                x: torch.Tensor, 
                y: torch.Tensor,
                edge_index: torch.Tensor,  
                node_idx: int, 
                forward_kwargs: dict = {}
            ) -> Tuple[dict, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        '''
        Get Guided Backpropagation explanation for one node in the graph
        Args:
            x (torch.tensor): tensor of node features from the entire graph
            node_idx (int): node index for which to explain a prediction around
            y (torch.Tensor): Ground truth labels correspond to each node's 
                classification. This argument is input to the `criterion` 
                function provided in `__init__()`.
            edge_index (torch.tensor): Edge_index of entire graph.
            forward_kwargs (dict, optional): Additional arguments to model.forward 
                beyond x and edge_index. Must be keyed on argument name. 
                (default: :obj:`{}`)

        :rtype: (:class:`dict`, (:class:`torch.Tensor`, :class:`torch.Tensor`, :class:`torch.Tensor`, :class:`torch.Tensor`))

        Returns:
            exp (dict):
                exp['feature'] (torch.Tensor, (s,k)): Explanations for each node, 
                    size `(s,k)` where `s` is number of nodes in the computational graph 
                    described around node `node_idx` and `k` is number of node input features. 
                exp['edge'] is `None` since there is no edge explanation generated.
            khop_info (4-tuple of torch.Tensor):
                0. the nodes involved in the subgraph
                1. the filtered `edge_index`
                2. the mapping from node indices in `node_idx` to their new location
                3. the `edge_index` mask indicating which edges were preserved  
        '''

        # Run whole-graph prediction:
        x.requires_grad = True

        # Perform the guided backprop:
        xhook = x.register_hook(clip_hook)
        
        self.model.zero_grad()
        pred = self.__forward_pass(x, edge_index, forward_kwargs)
        loss = self.criterion(pred, y)
        self.__apply_hooks()
        loss.backward()
        self.__rm_hooks()

        xhook.remove() # Remove hook from x

        graph_exp = x.grad

        khop_info = k_hop_subgraph(node_idx = node_idx, num_hops = self.L, edge_index = edge_index)
        subgraph_nodes = khop_info[0]

        exp = Explanation()
        # Get only those explanations for nodes in the subgraph:
        exp.node_imp = torch.stack([graph_exp[i,:] for i in subgraph_nodes])
        exp.node_idx = node_idx
        exp.set_whole_graph(x, edge_index)
        exp.set_enclosing_subgraph(khop_info)
        return exp

    def get_explanation_graph(self, 
                x: torch.Tensor, 
                y: torch.Tensor, 
                edge_index: torch.Tensor, 
                forward_kwargs: dict = {}
        ) -> dict:
        '''
        Explain a whole-graph prediction with Guided Backpropagation

        Args:
            x (torch.tensor): Tensor of node features from the entire graph.
            y (torch.tensor): Ground truth label of given input. This argument is 
                input to the `criterion` function provided in `__init__()`.
            edge_index (torch.tensor): Edge_index of entire graph.
            forward_kwargs (dict, optional): Additional arguments to model.forward 
                beyond x and edge_index. Must be keyed on argument name. 
                (default: :obj:`{}`)     

        :rtype: :class:`dict`
        
        Returns:
            exp (dict):
                exp['feature'] (torch.Tensor, (n,k)): Explanations for each node, 
                    size `(n,k)` where `n` is number of nodes in the entire graph 
                    described by `edge_index` and `k` is number of node input features. 
                exp['edge'] is `None` since there is no edge explanation generated.
        '''

        # Run whole-graph prediction:
        x.requires_grad = True

        # Perform the guided backprop:
        xhook = x.register_hook(clip_hook)
        
        self.model.zero_grad()
        pred = self.__forward_pass(x, edge_index, forward_kwargs)
        loss = self.criterion(pred, y)
        self.__apply_hooks()
        loss.backward()
        self.__rm_hooks()

        xhook.remove() # Remove hook from x

        exp = Explanation()
        exp.node_imp = x.grad
        exp.set_whole_graph(x, edge_index)

        #return {'feature': x.grad, 'edge': None}
        return exp

    def __apply_hooks(self):
        self.registered_hooks = []
        for p in self.model.parameters():
            h = p.register_hook(clip_hook)
            self.registered_hooks.append(h)

    def __rm_hooks(self):
        for h in self.registered_hooks:
            h.remove()
        self.registered_hooks = []
    
    def __forward_pass(self, x, edge_index, forward_kwargs):
        #@torch.enable_grad()
        # Forward pass:
        self.model.eval()
        self.__apply_hooks()
        pred = self.model(x, edge_index, **forward_kwargs)

        return pred

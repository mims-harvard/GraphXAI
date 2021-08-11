import torch
import torch.nn.functional as F
from typing import Tuple
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import k_hop_subgraph

import numpy as np

from ._decomp_base import _BaseDecomposition

class CAM(_BaseDecomposition):
    '''
    Class-Activation Mapping for GNNs
    '''

    def __init__(self, model: torch.nn.Module, activation = None):
        '''
        .. note::
            From Pope et al., CAM requires that the layer immediately before the softmax layer be
            a global average pooling layer, or in the case of node classification, a graph convolutional
            layer. Therefore, for this algorithm to theoretically work, there can be no fully-connected
            layers after global pooling. There is no restriction in the code for this, but be warned. 

        Args:
            model (torch.nn.Module): model on which to make predictions
            activation (method): activation funciton for final layer in network. If `activation = None`,
                explainer assumes linear activation. Use `activation = None` if the activation is applied
                within the `forward` method of `model`, only set this parameter if another activation is
                applied in the training procedure outside of model. 
        '''
        super().__init__(model=model)
        self.model = model

        # Set activation function
        self.activation = lambda x: x  if activation is None else activation
        # i.e. linear activation if none provided

    def get_explanation_node(self, 
                x: torch.Tensor, 
                node_idx: int, 
                edge_index: torch.Tensor, 
                label: int = None,  
                forward_kwargs: dict = {}
            ) -> Tuple[dict, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        '''
        Explain one node prediction by the model

        Args:
            x (torch.tensor): tensor of node features from the entire graph
            node_idx (int): node index for which to explain a prediction around
            edge_index (torch.tensor): edge_index of entire graph
            label (int, optional): Label on which to compute the explanation for
                this node. If `None`, the predicted label from the model will be
                used. (default: :obj:`None`)
            forward_kwargs (dict, optional): Additional arguments to model.forward 
                beyond x and edge_index. Must be keyed on argument name. 
                (default: :obj:`{}`)

        :rtype: (:class:`dict`, (:class:`torch.Tensor`, :class:`torch.Tensor`, :class:`torch.Tensor`, :class:`torch.Tensor`))

        Returns:
            exp (dict):
                exp['feature_imp'] is `None` because no feature mask is generated.
                exp['node_imp'] (torch.Tensor, (s,)): Explanations for each node, 
                    size `(s,)` where `s` is number of nodes in the computational
                    graph around node `node_idx`.
                exp['edge_imp'] is `None` since there is no edge explanation generated.
            khop_info (4-tuple of torch.Tensor):
                0. the nodes involved in the subgraph
                1. the filtered `edge_index`
                2. the mapping from node indices in `node_idx` to their new location
                3. the `edge_index` mask indicating which edges were preserved 
        '''
        label = int(self.__forward_pass(x, edge_index, forward_kwargs).argmax(dim=1).item()) if label is None else label

        # Perform walk:
        walk_steps, _ = self.extract_step(x, edge_index, detach=False, split_fc=False, forward_kwargs = forward_kwargs)

        #L = len(walk_steps)

        # Get subgraph:
        khop_info = k_hop_subgraph(node_idx = node_idx, num_hops = self.L, edge_index = edge_index)
        subgraph_nodes = khop_info[0]

        N = maybe_num_nodes(edge_index, None)
        subgraph_N = len(subgraph_nodes.tolist())

        #cam = torch.zeros(N) # Compute CAM only over the subgraph (all others are zero)
        cam = torch.zeros(subgraph_N)
        for i in range(subgraph_N):
            n = subgraph_nodes[i]
            cam[i] += self.__exp_node(n, walk_steps, label)

        return {'feature_imp': None, 'node_imp': cam, 'edge_imp': None}, khop_info

    def get_explanation_link(self, x, node_idx):
        raise NotImplementedError('Explanations for links is not implemented for Class-Activation Mapping')

    def get_explanation_graph(self, 
                x: torch.Tensor, 
                edge_index: torch.Tensor, 
                label: int = None, 
                num_nodes: int = None, 
                forward_kwargs: dict = {}
        ) -> dict:
        '''
        Explain a whole-graph prediction.

        Args:
            x (torch.tensor): tensor of node features from the entire graph
            edge_index (torch.tensor): edge_index of entire graph
            label (int, optional): Label on which to compute the explanation for
                this graph. If `None`, the predicted label from the model will be
                used. (default: :obj:`None`)
            num_nodes (int, optional): number of nodes in graph (default: :obj:`None`)
            forward_kwargs (dict, optional): Additional arguments to model.forward beyond x and edge_index. 
                Must be keyed on argument name. (default: :obj:`{}`)

        :rtype: :class:`dict`

        Returns:
            exp (dict):
                exp['feature_imp'] is `None` because no feature mask is generated.
                exp['node_imp'] (torch.Tensor, (n,)): Explanations for each node, 
                    size `(n,)` where `n` is number of nodes in the graph.
                exp['edge_imp'] is `None` since there is no edge explanation generated. 
        '''

        N = maybe_num_nodes(edge_index, num_nodes)

        # Forward pass:
        label = int(self.__forward_pass(x, edge_index, forward_kwargs).argmax(dim=1).item()) if label is None else label

        # Steps through model:
        walk_steps, _ = self.extract_step(x, edge_index, detach=True, split_fc=True, forward_kwargs = forward_kwargs)
        
        # Generate explanation for every node in graph
        node_explanations = []
        for n in range(N):
            node_explanations.append(self.__exp_node(n, walk_steps, label))

        return {'feature_imp': None, 'node_imp': torch.tensor(node_explanations), 'edge_imp': None}

    def __forward_pass(self, x, edge_index, forward_kwargs = {}):
        # Forward pass:
        self.model.eval()
        pred = self.model(x, edge_index, **forward_kwargs)

        return pred

    def __exp_node(self, node_idx, walk_steps, predicted_c):
        '''
        Gets explanation for one node
        Assumes ReLU activation after last convolutiuonal layer
        TODO: Fix activation function assumption
        '''
        last_conv_layer = walk_steps[-1]

        weight_vec = last_conv_layer['module'][0].weight[predicted_c, :].detach()
        F_l_n = F.relu(last_conv_layer['output'][node_idx,:]).detach()

        L_cam_n = F.relu(torch.matmul(weight_vec, F_l_n))

        return L_cam_n.item()


class Grad_CAM(_BaseDecomposition):
    '''
    Gradient Class-Activation Mapping for GNNs
    '''

    def __init__(self, model: torch.Tensor, criterion = F.cross_entropy):
        '''
        Args:
            model (torch.nn.Module): model on which to make predictions
            criterion (PyTorch Loss Function): loss function used to train the model.
                Needed to pass gradients backwards in the network to obtain gradients.
        '''
        super().__init__(model)
        self.model = model
        self.criterion = criterion

    def get_explanation_node(self, 
            x: torch.Tensor, 
            y: torch.Tensor, 
            node_idx: int, 
            edge_index: torch.Tensor, 
            label: int = None, 
            forward_kwargs: dict = {}, 
            average_variant: bool = True, 
            layer: int = 0
        ) -> Tuple[dict, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        '''
        Explain a node in the given graph
        Args:
            x (torch.Tensor, (n,)): Tensor of node features from the entire graph, with n nodes.
            y (torch.Tensor, (n,)): Ground-truth labels for all n nodes in the graph.
            node_idx (int): node index for which to explain a prediction around
            edge_index (torch.Tensor): edge_index of entire graph
            label (int, optional): Label for which to compute Grad-CAM against. If None, computes
                the Grad-CAM with respect to the model's predicted class for this node.
                (default :obj:`None`)
            forward_kwargs (dict, optional): additional arguments to model.forward 
                beyond x and edge_index. (default: :obj:`None`)
            average_variant (bool, optional): If True, computes the average Grad-CAM across all convolutional
                layers in the model. If False, computes Grad-CAM for `layer`. (default: :obj:`True`)
            layer (int, optional): Layer by which to compute the Grad-CAM. Argument only has an effect if 
                `average_variant == True`. Must be less-than the total number of convolutional layers
                in the model. (default: :obj:`0`)

        :rtype: (:class:`dict`, (:class:`torch.Tensor`, :class:`torch.Tensor`, :class:`torch.Tensor`, :class:`torch.Tensor`))

        Returns:
            exp (dict):
                exp['feature_imp'] is `None` because no feature mask is generated.
                exp['node_imp'] (torch.Tensor, (s,)): Explanations for each node, 
                    size `(s,)` where `s` is number of nodes in computational graph
                    around node `node_idx`.
                exp['edge_imp'] is `None` since there is no edge explanation generated.
            khop_info (4-tuple of torch.Tensor):
                0. the nodes involved in the subgraph
                1. the filtered `edge_index`
                2. the mapping from node indices in `node_idx` to their new location
                3. the `edge_index` mask indicating which edges were preserved 
        '''

        x.requires_grad = True

        if label is None:
            pred = self.__forward_pass(x, y, edge_index, forward_kwargs)[0][node_idx, :].reshape(1, -1)
            y[node_idx] = pred.argmax(dim=1).item()
        else: # Transform node_idx's label if provided by user
            pred, loss = self.__forward_pass(x, y, edge_index, forward_kwargs)
            y[node_idx] = label

        walk_steps, _ = self.extract_step(x, edge_index, detach=True, split_fc=True, forward_kwargs = forward_kwargs)

        #L = len(walk_steps)

        khop_info = k_hop_subgraph(node_idx, self.L, edge_index)
        subgraph_nodes = khop_info[0]

        N = maybe_num_nodes(edge_index, None)
        subgraph_N = len(subgraph_nodes.tolist())

        if average_variant:
            # Size of all nodes in graph; Sets all values to zero if not in subgraph
            #avg_gcam = torch.zeros(N)
            avg_gcam = torch.zeros(subgraph_N)

            for l in range(self.L):
                # Compute gradients for this layer ahead of time:
                gradients = self.__grad_by_layer(l)

                for i in range(subgraph_N): # Over all subgraph nodes
                    n = subgraph_nodes[i]
                    avg_gcam[i] += self.__get_gCAM_layer(walk_steps, l, n, gradients)#[0]

            avg_gcam /= self.L # Apply average

            return {'feature_imp': None, 'node_imp': avg_gcam, 'edge_imp': None}, khop_info

        else:
            assert layer < len(walk_steps), "Layer must be an index of convolutional layers"

            gcam = torch.zeros(subgraph_N)
            gradients = self.__grad_by_layer(layer)
            for i in range(subgraph_N):
                n = subgraph_nodes[i]
                gcam[i] += self.__get_gCAM_layer(walk_steps, layer, n, gradients)#[0]

            return {'feature_imp': None, 'node_imp': gcam, 'edge_imp': None}, khop_info
                
    def get_explanation_link(self, x, node_idx):
        raise NotImplementedError('Explanations for links is not implemented for Gradient Class-Activation Mapping')

    def get_explanation_graph(self, 
                x: torch.Tensor, 
                edge_index: torch.Tensor, 
                label: int = None, 
                num_nodes: int = None, 
                forward_kwargs: dict = {}, 
                average_variant: bool = True, 
                layer: int = 0
        ) -> dict:
        '''
        Explain a whole-graph prediction.

        Args:
            x (torch.tensor): tensor of node features from the entire graph
            edge_index (torch.tensor): edge_index of entire graph
            label (int, optional): Label for which to compute Grad-CAM against. If None, computes the 
                Grad-CAM with respect to the model's predicted class for this node. (default :obj:`None`)
            num_nodes (int, optional): number of nodes in graph (default: :obj:`None`)
            forward_kwargs (dict, optional): Additional arguments to model.forward beyond x and edge_index. 
                Must be keyed on argument name. (default: :obj:`{}`)
            average_variant (bool, optional): If True, computes the average Grad-CAM across all convolutional
                layers in the model. If False, computes Grad-CAM for `layer`. (default: :obj:`True`)
            layer (int, optional): Layer by which to compute the Grad-CAM. Argument only has an effect if 
                `average_variant == True`. Must be less-than the total number of convolutional layers
                in the model. (default: :obj:`0`)

        :rtype: :class:`dict`
        
        Returns:
            exp (dict):
                exp['feature_imp'] is `None` because no feature mask is generated.
                exp['node_imp'] (torch.Tensor, (n,)): Explanations for each node, 
                    size `(n,)` where `n` is number of nodes in the graph.
                exp['edge_imp'] is `None` since there is no edge explanation generated.
        '''

        self.N = maybe_num_nodes(edge_index, num_nodes)

        if label is None:
            self.model.eval()
            pred = self.model(x, edge_index, **forward_kwargs)
            label = pred.argmax(dim=1).item()

        # Execute forward pass:
        pred, loss = self.__forward_pass(x, torch.tensor([label], dtype=torch.long), edge_index, forward_kwargs)

        # Calculate predicted class and steps through model:
        walk_steps, _ = self.extract_step(x, edge_index, detach=True, split_fc=True, forward_kwargs = forward_kwargs)
        
        # Generate explanation for every node in graph
        if average_variant:
            avg_gcam = torch.zeros(self.N)

            for l in range(self.L): # Get Grad
                avg_gcam += np.array(self.__get_gCAM_layer(walk_steps, layer=l))

            # Element-wise average over all nodes:
            avg_gcam /= self.L
            return {'feature_imp': None, 'node_imp': avg_gcam, 'edge_imp': None}

        else:
            assert layer < len(walk_steps), "Layer must be an index of convolutional layers"

            return {'feature_imp': None, 
                'node_imp': torch.tensor(self.__get_gCAM_layer(walk_steps, layer=layer)), 
                'edge_imp': None}

    def __forward_pass(self, x, label, edge_index, forward_kwargs):
        x.requires_grad = True # Enforce that x needs gradient

        # Forward pass:
        self.model.eval()
        pred = self.model(x, edge_index, **forward_kwargs)

        loss = self.criterion(pred, label)
        loss.backward() # Propagate loss backward through network

        return pred, loss

    def __grad_by_layer(self, layer):
        # Index 0 of parameters to avoid calculating gradients for biases
        return list(list(self.model.children())[layer].parameters())[0].grad.mean(dim=0)

    def __get_gCAM_layer(self, walk_steps, layer, node_idx = None, gradients = None):
        # Gets Grad CAM for one layer
        if gradients is None:
            # \alpha^{l,c} = Average over nodes of gradients for layer l, after activation over c
            # ''        '' shape: [k,] - k= # output features from layer l
            gradients = self.__grad_by_layer(layer)

        if node_idx is None: # Need to compute for entire graph:
            node_explanations = []
            for n in range(self.N):
                node_explanations.append(self.__exp_node(n, walk_steps, layer, gradients))

            return node_explanations

        # Return for only one node:
        return self.__exp_node(node_idx, walk_steps, layer, gradients)

    def __exp_node(self, node_idx, walk_steps, layer, gradients):
        '''
        Gets explanation for one node
        Assumes ReLU activation after each convolutional layer
        TODO: Fix activation function assumption
        '''
        # \alpha^{l,c} = Average over nodes of gradients for layer l, after activation over c
        # ''        '' shape: [k,] - k= # input features to layer l

        # Activations for node n
        F_l_n = F.relu(walk_steps[layer]['output'][node_idx,:]).detach()
        L_cam_n = F.relu(torch.matmul(gradients, F_l_n)) # Combine gradients and activations

        return L_cam_n.item()

import torch
import torch.nn.functional as F
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import k_hop_subgraph

import numpy as np

from .utils.base_explainer import WalkBase

class CAM(WalkBase):
    '''
    Class-Activation Mapping for GNNs
    '''

    def __init__(self, model: torch.nn.Module, device: str, activation = None):
        '''
        Args:
            model (torch.nn.Module): model on which to make predictions
            device: device on which model is to run
            activation (method): activation funciton for final layer in network. If `activation = None`,
                explainer assumes linear activation. Use `activation = None` if the activation is applied
                within the `forward` method of `model`, only set this parameter if another activation is
                applied in the training procedure outside of model. 
        '''
        super(WalkBase, self).__init__(model)
        self.model = model
        self.device = device

        if activation is None:
            self.activation = lambda x: x # i.e. linear activation
        else:
            self.activation = activation # Set activation function

    def get_explanation_node(self, x: torch.Tensor, node_idx: int, edge_index: torch.Tensor, 
        forward_args: tuple = None):
        '''
        Explain one node prediction by the model
        Args:
            x (torch.tensor): tensor of node features from the entire graph
            node_idx (int): node index for which to explain a prediction around
            edge_index (torch.tensor): edge_index of entire graph
            forward_args (tuple): additional arguments to model.forward 
                beyond x and edge_index. (default: :obj:`None`)
        '''
        pred = self.__forward_pass(x, edge_index, forward_args)[node_idx, :].reshape(1, -1)
        predicted_c = self.activation(pred)

        # Perform walk:
        walk_steps, fc_steps = self.extract_step(x, edge_index, detach=False, split_fc=False, forward_args = forward_args)

        L = len(walk_steps)

        # Get subgraph:
        khop_info = k_hop_subgraph(node_idx = node_idx, num_hops = L, edge_index = edge_index)
        subgraph_nodes = khop_info[0]
        subgraph_eidx = khop_info[1]

        N = maybe_num_nodes(edge_index, None)
        subgraph_N = len(subgraph_nodes.tolist())

        cam = np.zeros(N) # Compute CAM only over the subgraph (all others are zero)
        for i in range(subgraph_N):
            n = subgraph_nodes[i]
            cam[n] += self.__exp_node(n, walk_steps, predicted_c)

        return cam, khop_info

    def get_explanation_link(self, x, node_idx):
        raise NotImplementedError('Explanations for links is not implemented for Class-Activation Mapping')

    def get_explanation_graph(self, x: torch.Tensor, edge_index: torch.Tensor, num_nodes: int = None, 
        forward_args: tuple = None):
        '''
        Explain a whole-graph prediction.

        Args:
            x (torch.tensor): tensor of node features from the entire graph
            edge_index (torch.tensor): edge_index of entire graph
            num_nodes (int, optional): number of nodes in graph (default: :obj:`None`)
            forward_args (tuple, optional): additional arguments to model.forward 
                beyond x and edge_index. (default: :obj:`None`)
        '''

        N = maybe_num_nodes(edge_index, num_nodes)

        # Forward pass:
        self.model.eval()
        if forward_args is None:
            pred = self.model(x, edge_index)
        else:
            pred = self.model(x, edge_index, *forward_args)

        # Calculate predicted class and steps through model:
        predicted_c = self.activation(pred)
        walk_steps, _ = self.extract_step(x, edge_index, detach=True, split_fc=True, forward_args = forward_args)
        
        # Generate explanation for every node in graph
        node_explanations = []
        for n in range(N):
            node_explanations.append(self.__exp_node(n, walk_steps, predicted_c))

        return node_explanations

    def __forward_pass(self, x, edge_index, forward_args):
        # Forward pass:
        self.model.eval()
        if forward_args is None:
            pred = self.model(x, edge_index)
        else:
            pred = self.model(x, edge_index, *forward_args)

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


class Grad_CAM(WalkBase):
    '''
    Gradient Class-Activation Mapping for GNNs
    '''

    def __init__(self, model: torch.Tensor, device: str, criterion = F.cross_entropy):
        '''
        Args:
            model (torch.nn.Module): model on which to make predictions
            device: device on which model is to run
            criterion (PyTorch Loss Function): loss function used to train the model.
                Needed to pass gradients backwards in the network to obtain gradients.
        '''
        super(WalkBase, self).__init__(model)
        self.model = model
        self.device = device
        self.criterion = criterion

    def get_explanation_node(self, x: torch.Tensor, labels: torch.Tensor, node_idx: int, 
        edge_index: torch.Tensor, node_idx_label: int = None, forward_args: tuple = None, 
        average_variant: bool = True, layer: int = 0):
        '''
        Explain a node in the given graph
        Args:
            x (torch.tensor): tensor of node features from the entire graph
            node_idx (int): node index for which to explain a prediction around
            edge_index (torch.tensor): edge_index of entire graph
            node_idx_label (int, optional): label for which to compute Grad-CAM against. If None, computes
                the Grad-CAM with respect to the model's predicted class for this node.
                (default :obj:`None`)
            forward_args (tuple, optional): additional arguments to model.forward 
                beyond x and edge_index. (default: :obj:`None`)
            average_variant (bool, optional): If True, computes the average Grad-CAM across all convolutional
                layers in the model. If False, computes Grad-CAM for `layer`. (default: :obj:`True`)
            layer (int, optional): Layer by which to compute the Grad-CAM. Argument only has an effect if 
                `average_variant == True`. Must be less-than the total number of convolutional layers
                in the model. (default: :obj:`0`)
        '''

        x.requires_grad = True

        if node_idx_label is not None: # Transform node_idx's label if provided by user
            labels[node_idx] = node_idx_label

        pred, loss = self.__forward_pass(x, labels, edge_index, forward_args)#[node_idx, :].reshape(1, -1)

        walk_steps, fc_steps = self.extract_step(x, edge_index, detach=True, split_fc=True, forward_args = forward_args)

        L = len(walk_steps)

        khop_info = k_hop_subgraph(node_idx, L, edge_index)
        subgraph_nodes = khop_info[0]
        subgraph_eidx = khop_info[1]

        N = maybe_num_nodes(edge_index, None)
        subgraph_N = len(subgraph_nodes.tolist())

        if average_variant:
            # Size of all nodes in graph; Sets all values to zero if not in subgraph
            avg_gcam = np.zeros(N)

            for l in range(L):
                # Compute gradients for this layer ahead of time:
                gradients = self.__grad_by_layer(l)

                for i in range(subgraph_N): # Over all subgraph nodes
                    n = subgraph_nodes[i]
                    avg_gcam[n] += self.__get_gCAM_layer(walk_steps, l, n, gradients)#[0]

            avg_gcam /= L # Apply average

            return avg_gcam, khop_info

        else:
            assert layer < len(walk_steps), "Layer must be an index of convolutional layers"

            gcam = np.zeros(N)
            gradients = self.__grad_by_layer(layer)
            for i in range(subgraph_N):
                n = subgraph_nodes[i]
                gcam[n] += self.__get_gCAM_layer(walk_steps, layer, n, gradients)#[0]

            return gcam, khop_info
                

    def get_explanation_link(self, x, node_idx):
        raise NotImplementedError('Explanations for links is not implemented for Gradient Class-Activation Mapping')

    def get_explanation_graph(self, x: torch.Tensor, label: int, edge_index: torch.Tensor, num_nodes: int = None, 
        forward_args: tuple = None, average_variant: bool = True, layer: int = 0):
        '''
        Explain a whole-graph prediction.

        Args:
            x (torch.tensor): tensor of node features from the entire graph
            label (int): label for which to compute Grad-CAM against
            edge_index (torch.tensor): edge_index of entire graph
            num_nodes (int, optional): number of nodes in graph (default: :obj:`None`)
            forward_args (tuple, optional): additional arguments to model.forward 
                beyond x and edge_index. (default: :obj:`None`)
            average_variant (bool, optional): If True, computes the average Grad-CAM across all convolutional
                layers in the model. If False, computes Grad-CAM for `layer`. (default: :obj:`True`)
            layer (int, optional): Layer by which to compute the Grad-CAM. Argument only has an effect if 
                `average_variant == True`. Must be less-than the total number of convolutional layers
                in the model. (default: :obj:`0`)
        '''

        self.N = maybe_num_nodes(edge_index, num_nodes)

        # Execute forward pass:
        pred, loss = self.__forward_pass(x, torch.tensor([label], dtype=torch.long), edge_index, forward_args)

        # Calculate predicted class and steps through model:
        walk_steps, fc_steps = self.extract_step(x, edge_index, detach=True, split_fc=True, forward_args = forward_args)
        
        # Generate explanation for every node in graph
        if average_variant:
            avg_gcam = np.zeros(self.N)

            for l in range(len(walk_steps)): # Get Grad
                avg_gcam += np.array(self.__get_gCAM_layer(walk_steps, layer=l))

            # Element-wise average over all nodes:
            avg_gcam /= len(walk_steps)
            return avg_gcam

        else:
            assert layer < len(walk_steps), "Layer must be an index of convolutional layers"

            return self.__get_gCAM_layer(walk_steps, layer=layer)

    def __forward_pass(self, x, label, edge_index, forward_args):
        x.requires_grad = True # Enforce that x needs gradient

        # Forward pass:
        self.model.eval()
        if forward_args is None:
            pred = self.model(x, edge_index)
        else:
            pred = self.model(x, edge_index, *forward_args)

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
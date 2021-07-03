import torch
import torch.nn.functional as F
#from torch_geometric import global_mean_pool
from torch_geometric.utils.num_nodes import maybe_num_nodes

from .utils.base_explainer import WalkBase

class CAM(WalkBase):
    '''
    Class-Activation Mapping for GNNs
    '''

    def __init__(self, model, device):
        super(WalkBase, self).__init__(model)
        self.model = model
        self.device = device

    def get_explanation_node(self, x, node_idx, edge_index, forward_args = None):
        #raise NotImplementedError('Explanations for nodes is not implemented for Class-Activation Mapping')
        '''
        TODO: Change to whole-graph explanation
        '''

        # Forward pass:
        self.model.eval()
        if forward_args is None:
            pred = self.model(x, edge_index)
        else:
            pred = self.model(x, edge_index, *forward_args)

        predicted_c = pred.argmax(dim=1)

        # Perform walk:
        walk_steps, _ = self.extract_step(x, edge_index, detach=False, split_fc=True, forward_args = forward_args)

        # Call private function:
        return self.__exp_node(node_idx, walk_steps, predicted_c)

    def get_explanation_link(self, x, node_idx):
        raise NotImplementedError('Explanations for links is not implemented for Class-Activation Mapping')

    def get_explanation_graph(self, x, edge_index, num_nodes = None, forward_args = None):

        N = maybe_num_nodes(edge_index, num_nodes)

        # Forward pass:
        self.model.eval()
        if forward_args is None:
            pred = self.model(x, edge_index)
        else:
            pred = self.model(x, edge_index, *forward_args)

        # Calculate predicted class and steps through model:
        predicted_c = pred.argmax(dim=1)
        walk_steps, _ = self.extract_step(x, edge_index, detach=True, split_fc=True, forward_args = forward_args)
        
        # Generate explanation for every node in graph
        node_explanations = []
        for n in range(N):
            node_explanations.append(self.__exp_node(n, walk_steps, predicted_c))

        return node_explanations

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

    def __init__(self, model, device):
        super(WalkBase, self).__init__(model)
        self.model = model
        self.device = device

    def get_explanation_node(self, x, label, node_idx, edge_index, layer = 0, num_nodes = None, 
                            forward_args = None, criterion = F.cross_entropy):
        '''
        TODO: Change to whole-graph explanation
        label (int): label for given input.
            Not restricted to actual label of input, can be any label.
        '''

        N = maybe_num_nodes(edge_index, num_nodes)
        x.requires_grad = True # Enforce that x needs gradient

        # Forward pass:
        self.model.eval()
        if forward_args is None:
            pred = self.model(x, edge_index)
        else:
            pred = self.model(x, edge_index, *forward_args)

        loss = criterion(pred, label)
        loss.backward() # Propagate loss backward through network

        # Perform walk:
        walk_steps, _ = self.extract_step(x, edge_index, detach=False, split_fc=True, forward_args = forward_args)

        assert layer < len(walk_steps), "Layer must be an index of convolutional layers"

        # \alpha^{l,c} = Average over nodes of gradients for layer l, after activation over c
        # ''        '' shape: [k,] - k= # input features to layer l
        gradients = list(self.model.parameters())[layer].grad.sum(dim=0) / N

        # Call private function:
        return self.__exp_node(node_idx, walk_steps, layer, gradients)

    def get_explanation_link(self, x, node_idx):
        raise NotImplementedError('Explanations for links is not implemented for Class-Activation Mapping')

    def get_explanation_graph(self, x, label, edge_index, layer = 0, num_nodes = None, forward_args = None, 
                        criterion = F.cross_entropy):

        N = maybe_num_nodes(edge_index, num_nodes)

        x.requires_grad = True # Enforce that x needs gradient

        # Forward pass:
        self.model.eval()
        if forward_args is None:
            pred = self.model(x, edge_index)
        else:
            pred = self.model(x, edge_index, *forward_args)

        loss = criterion(pred, label)
        loss.backward() # Propagate loss backward through network

        # Calculate predicted class and steps through model:
        predicted_c = pred.argmax(dim=1)
        walk_steps, _ = self.extract_step(x, edge_index, detach=True, split_fc=True, forward_args = forward_args)

        assert layer < len(walk_steps), "Layer must be an index of convolutional layers"

        # \alpha^{l,c} = Average over nodes of gradients for layer l, after activation over c
        # ''        '' shape: [k,] - k= # input features to layer l
        gradients = list(self.model.parameters())[layer].grad.sum(dim=0) / N

        # Gradient averaging is same for every node
        
        # Generate explanation for every node in graph
        node_explanations = []
        for n in range(N):
            node_explanations.append(self.__exp_node(n, walk_steps, layer, gradients))

        return node_explanations

    def __exp_node(self, node_idx, walk_steps, layer, gradients):
        '''
        Gets explanation for one node
        Assumes ReLU activation after last convolutiuonal layer
        TODO: Fix activation function assumption
        '''
        # \alpha^{l,c} = Average over nodes of gradients for layer l, after activation over c
        # ''        '' shape: [k,] - k= # input features to layer l

        # Activations for node n
        F_l_n = F.relu(walk_steps[layer]['output'][node_idx,:]).detach()
        L_cam_n = F.relu(torch.matmul(gradients, F_l_n)) # Combine gradients and activations

        return L_cam_n.item()
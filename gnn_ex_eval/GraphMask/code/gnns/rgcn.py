from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot, uniform
import torch
from torch.nn import Linear, ReLU, Sequential, LayerNorm
from torch.nn import Parameter
import numpy as np

from code.abstract.abstract_gnn import AbstractGNN


class RGCNLayer(MessagePassing):

    """
    A simple implementation of R-GCN message passing and aggregation used for the synthetic task.
    Modified slightly from regular pytorch-geometric to allow scaling and replacement of messages.
    """

    def __init__(self,
                 in_dim,
                 out_dim,
                 num_relations):
        super(RGCNLayer, self).__init__('add')

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_relations = num_relations

        self.basis = Parameter(torch.Tensor(in_dim, out_dim * num_relations))
        self.bias = Parameter(torch.Tensor(num_relations, out_dim))

        self.residual = Sequential(Linear(in_dim + out_dim, out_dim),
                                   LayerNorm(out_dim),
                                   ReLU())

        self.process_message = Sequential(LayerNorm(out_dim),
                                          ReLU())

        self.reset_parameters()

    def reset_parameters(self):
        size = self.num_relations * self.in_dim

        glorot(self.basis)
        uniform(size, self.bias)

    def forward(self, x, edge_index, edge_type, message_scale=None, message_replacement=None):
        """"""
        size = [x.shape[0], x.shape[0]]

        res = self.propagate(
            edge_index, x=x, edge_type=edge_type, size=size, message_scale=message_scale,
            message_replacement=message_replacement)

        return res

    def message(self, x_j, x_i, edge_type, message_scale, message_replacement):
        b = torch.index_select(self.bias, 0, edge_type)

        basis_messages = torch.matmul(x_j, self.basis).view(-1, self.bias.shape[0], self.out_dim)
        count = torch.arange(edge_type.shape[0])
        basis_messages = basis_messages[count, edge_type, :] + b

        basis_messages = self.process_message(basis_messages)

        if message_scale is not None:
            basis_messages = basis_messages * message_scale.unsqueeze(-1)

            if message_replacement is not None:
                if basis_messages.shape == message_replacement.shape:
                    basis_messages = basis_messages + (1 - message_scale).unsqueeze(-1) * message_replacement
                else:
                    basis_messages = basis_messages + (1 - message_scale).unsqueeze(-1) * message_replacement.unsqueeze(
                        0)

        self.latest_messages = basis_messages
        self.latest_source_embeddings = x_j
        self.latest_target_embeddings = x_i

        return basis_messages

    def get_latest_source_embeddings(self):
        return self.latest_source_embeddings

    def get_latest_target_embeddings(self):
        return self.latest_target_embeddings

    def get_latest_messages(self):
        return self.latest_messages

    def update(self, aggr_out, x):
        repr = torch.cat((aggr_out, x), 1)
        return self.residual(repr)


class RGCN(AbstractGNN):

    def __init__(self,input_dim,output_dim,n_relations,n_layers,inverse_edges=False,separate_relation_types_for_inverse=False):
        AbstractGNN.__init__(self)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_relations = n_relations
        self.n_layers = n_layers
        self.inverse_edges = inverse_edges
        self.separate_relation_types_for_inverse = separate_relation_types_for_inverse

        self.define_weights_and_layers()

    def is_adj_mat(self):
        return False

    def define_weights_and_layers(self):
        gnn_layers = []

        use_rels = self.n_relations
        if self.inverse_edges and self.separate_relation_types_for_inverse:
            use_rels *= 2

        for layer in range(self.n_layers):
            gnn_layers.append(RGCNLayer(self.output_dim, self.output_dim, use_rels))

        gnn_layers = torch.nn.ModuleList(gnn_layers)
        self.gnn_layers = gnn_layers
        self.W_input = torch.nn.Sequential(Linear(self.input_dim, self.output_dim),
                                           LayerNorm(self.output_dim),
                                           ReLU())

    def set_device(self, device):
        pass

    def get_initial_layer_input(self, vertex_embeddings):
        return self.W_input(vertex_embeddings)

    def process_layer(self, vertex_embeddings, edges, edge_types, gnn_layer, message_scale, message_replacement, edge_direction_cutoff=None):
        layer_output = gnn_layer(vertex_embeddings,
                                 edges,
                                 edge_types,
                                 message_scale=message_scale,
                                 message_replacement=message_replacement)

        return layer_output


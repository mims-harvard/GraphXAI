import torch
from torch.nn import Linear, ReLU, Sequential, LayerNorm, Sigmoid, Dropout, Module

from code.abstract.abstract_adj_mat_gnn import AbstractAdjMatGNN
from code.utils.torch_utils.xavier_linear import XavierLinear


class QaGnnMessagePasser(Module):
    """
    Edge types are few enough that we can afford a full W_r for every type.
    """

    def __init__(self,
                 channels,
                 num_relations):
        super(QaGnnMessagePasser, self).__init__()

        self.channels = channels
        self.num_relations = num_relations

        self.w = Sequential(XavierLinear(channels, channels * (self.num_relations + 1)))

        self.att_w = Sequential(XavierLinear(channels * 2, channels),
                                Sigmoid())

        self.dropout = Dropout(0.2)

    def get_in_dim(self):
        return self.channels

    def get_out_dim(self):
        return self.channels

    def forward(self, x, adj_mat, mask, message_scale, message_replacement):
        self.latest_vertex_embeddings = x

        h = self.w(x).view(adj_mat.shape[0], -1, self.num_relations + 1, self.channels) * mask.unsqueeze(2)

        h_msg = h[:, :, :self.num_relations, :].transpose(1, 2)
        h_self = h[:, :, self.num_relations, :]

        if message_scale is not None:
            gated_adj_mat = message_scale * adj_mat
            msg = torch.matmul(gated_adj_mat, h_msg)

            if message_replacement is not None:
                inv_gated_adj_mat = (1 - message_scale) * adj_mat
                message_replacement = message_replacement.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(x.size()[0],
                                                                                                        self.num_relations,
                                                                                                        x.size()[1], 1)
                replacement_msg = torch.matmul(inv_gated_adj_mat, message_replacement)

                msg += replacement_msg
        else:
            msg = torch.matmul(adj_mat, h_msg)

        msg = msg.sum(dim=1)
        msg += h_self

        att_input = torch.cat([x, msg], dim=-1)
        att = self.att_w(att_input)

        update = att * torch.tanh(msg) + (1 - att) * x
        update = self.dropout(update)

        return update

    def get_latest_vertex_embeddings(self):
        return self.latest_vertex_embeddings


class QaGNN(AbstractAdjMatGNN):

    def __init__(self, dim, n_layers, n_relations, share_parameters=True):
        super(QaGNN, self).__init__()

        self.dim = dim
        self.n_layers = n_layers
        self.n_relations = n_relations

        gnn_layers = []
        for layer in range(self.n_layers):
            gnn_layers.append(QaGnnMessagePasser(self.dim, self.n_relations))

        if share_parameters: # This is a bit of a hack, but we need separate objects with the same weight tensors
            gnn_0 = gnn_layers[0]
            for gnn_layer in gnn_layers:
                gnn_layer.w = gnn_0.w
                gnn_layer.att_w = gnn_0.att_w

        gnn_layers = torch.nn.ModuleList(gnn_layers)
        self.gnn_layers = gnn_layers

    def is_adj_mat(self):
        return True

    def get_initial_layer_input(self, vertex_embeddings, mask):
        return vertex_embeddings * mask

    def process_layer(self, vertex_embeddings,
                      adj_mat,
                      gnn_layer,
                      message_scale,
                      message_replacement,
                      mask):

        return gnn_layer(vertex_embeddings, adj_mat, mask, message_scale, message_replacement)

    def process_output(self, layer_input, mask):
        return layer_input * mask
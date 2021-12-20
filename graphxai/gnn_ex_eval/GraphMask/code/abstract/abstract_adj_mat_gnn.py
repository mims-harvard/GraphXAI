import torch
import numpy as np


class AbstractAdjMatGNN(torch.nn.Module):
    injected_message_scale = None
    injected_message_replacement = None
    latest_adj_mat = None

    def __init__(self):
        torch.nn.Module.__init__(self)

    def get_initial_layer_input(self, vertex_embeddings):
        return vertex_embeddings

    def inject_message_scale(self, message_scale):
        self.injected_message_scale = message_scale

    def inject_message_replacement(self, message_replacement):
        self.injected_message_replacement = [
            message_replacement]  #inject_message_scale Have to store it in a list to prevent the pytorch module from thinking it is a parameter

    def get_vertex_embedding_dims(self):
        return np.array([layer.get_in_dim() for layer in self.gnn_layers])

    def get_latest_vertex_embeddings(self):
        return [layer.get_latest_vertex_embeddings() for layer in self.gnn_layers]

    def get_latest_adj_mat(self):
        return self.latest_adj_mat

    def get_message_dims(self):
        return np.array([layer.get_out_dim() for layer in self.gnn_layers])

    def count_latest_messages(self):
        return self.latest_adj_mat.count_nonzero().detach().cpu().numpy() * self.count_layers()

    def count_layers(self):
        return self.n_layers

    def forward(self, vertex_embeddings, adj_mat, mask=None):
        layer_input = self.get_initial_layer_input(vertex_embeddings)

        self.latest_adj_mat = adj_mat

        for i, gnn_layer in enumerate(self.gnn_layers):
            if self.injected_message_scale is not None:
                message_scale = self.injected_message_scale[i]
            else:
                message_scale = None

            if self.injected_message_replacement is not None:
                message_replacement = self.injected_message_replacement[0][i]
            else:
                message_replacement = None

            layer_input = self.process_layer(vertex_embeddings=layer_input,
                                             adj_mat=adj_mat,
                                             mask=mask,
                                             gnn_layer=gnn_layer,
                                             message_scale=message_scale,
                                             message_replacement=message_replacement)

        output = layer_input

        if self.injected_message_scale is not None:
            self.injected_message_scale = None

        if self.injected_message_replacement is not None:
            self.injected_message_replacement = None

        return output

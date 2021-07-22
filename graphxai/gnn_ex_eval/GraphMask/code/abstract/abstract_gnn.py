import torch
import numpy as np

class AbstractGNN(torch.nn.Module):

    injected_message_scale = None
    injected_message_replacement = None

    def __init__(self):
        torch.nn.Module.__init__(self)

    def get_initial_layer_input(self, vertex_embeddings):
        return vertex_embeddings

    def inject_message_scale(self, message_scale):
        self.injected_message_scale = message_scale

    def inject_message_replacement(self, message_replacement):
        self.injected_message_replacement = [message_replacement] # Have to store it in a list to prevent the pytorch module from thinking it is a parameter

    def get_vertex_embedding_dims(self):
        return np.array([layer.in_dim for layer in self.gnn_layers])

    def get_message_dims(self):
        return np.array([layer.out_dim for layer in self.gnn_layers])

    def get_latest_source_embeddings(self):
        return [layer.get_latest_source_embeddings() for layer in self.gnn_layers]

    def get_latest_target_embeddings(self):
        return [layer.get_latest_target_embeddings() for layer in self.gnn_layers]

    def get_latest_messages(self):
        return [layer.get_latest_messages() for layer in self.gnn_layers]

    def count_latest_messages(self):
        return sum([layer_messages.numel()/layer_messages.shape[-1] for layer_messages in self.get_latest_messages()])

    def count_layers(self):
        return self.n_layers

    def forward(self, vertex_embeddings, edges, edge_types, edge_direction_cutoff=None):
        layer_input = self.get_initial_layer_input(vertex_embeddings)

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
                                             edges=edges,
                                             edge_types=edge_types,
                                             gnn_layer=gnn_layer,
                                             edge_direction_cutoff=edge_direction_cutoff,
                                             message_scale=message_scale,
                                             message_replacement=message_replacement)

        output = layer_input

        if self.injected_message_scale is not None:
            self.injected_message_scale = None

        if self.injected_message_replacement is not None:
            self.injected_message_replacement = None

        return output
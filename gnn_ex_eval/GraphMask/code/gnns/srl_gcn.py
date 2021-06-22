import torch
from torch.nn import Linear, ReLU, Sigmoid, Tanh, Dropout, LayerNorm, ELU, Embedding
from torch_geometric.nn import MessagePassing

from code.abstract.abstract_gnn import AbstractGNN


class SrlGcnMessagePasser(MessagePassing):

    has_hard_gates = False

    def __init__(self,
                 in_dim,
                 out_dim,
                 n_relations):
        super(SrlGcnMessagePasser, self).__init__('add', flow="source_to_target")

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_relations = n_relations

        self.W_message_forward = Linear(self.in_dim, self.out_dim, bias=False)
        self.W_message_backward = Linear(self.in_dim, self.out_dim, bias=False)
        self.W_gate_forward = Linear(self.in_dim, 1, bias=False)
        self.W_gate_backward = Linear(self.in_dim, 1, bias=False)

        self.b_message_foward = Embedding(self.n_relations, self.out_dim)
        self.b_message_backward = Embedding(self.n_relations, self.out_dim)
        self.b_gate_foward = Embedding(self.n_relations, 1)
        self.b_gate_backward = Embedding(self.n_relations, 1)

        self.W_self_loop = Linear(self.in_dim, self.out_dim)
        self.W_gate_self_loop = Linear(self.in_dim, 1)

    def forward(self, vertex_embeddings, edge_direction_cutoff, edge_types, edges, message_scale=None, message_replacement=None):
        size = [vertex_embeddings.shape[0], vertex_embeddings.shape[0]]

        new_vertex_embeddings = self.propagate(
            edges,
            x=vertex_embeddings,
            edge_direction_cutoff=edge_direction_cutoff,
            edge_types=edge_types,
            size=size,
            message_scale=message_scale,
            message_replacement=message_replacement
        )

        return new_vertex_embeddings

    def message(self, x_j, x_i, edge_direction_cutoff, edge_types, message_scale, message_replacement):
        if x_j.shape[0] == 0:
            return torch.empty((0, x_j.shape[1])).to(self.device)

        h_m_forward = self.W_message_forward(x_j[:edge_direction_cutoff])
        h_m_backward = self.W_message_backward(x_j[edge_direction_cutoff:])
        h_g_forward = self.W_gate_forward(x_j[:edge_direction_cutoff])
        h_g_backward = self.W_gate_backward(x_j[edge_direction_cutoff:])

        b_m_forward = self.b_message_foward(edge_types[:edge_direction_cutoff])
        b_m_backward = self.b_message_backward(edge_types[edge_direction_cutoff:])
        b_g_forward = self.b_gate_foward(edge_types[:edge_direction_cutoff])
        b_g_backward = self.b_gate_backward(edge_types[edge_direction_cutoff:])

        g_forward = torch.sigmoid(h_g_forward + b_g_forward)
        g_backward = torch.sigmoid(h_g_backward + b_g_backward)

        m_forward = g_forward * (h_m_forward + b_m_forward)
        m_backward = g_backward * (h_m_backward + b_m_backward)

        updates = torch.cat([m_forward, m_backward], dim=0)

        if message_scale is not None:
            updates = updates * message_scale.unsqueeze(-1)

            if message_replacement is not None:
                updates = updates + (1 - message_scale).unsqueeze(-1) * message_replacement.unsqueeze(0)

        self.latest_messages = updates
        self.latest_source_embeddings = x_j
        self.latest_target_embeddings = x_i

        return updates

    def update(self, aggr_out, x):
        m_self_loop = self.W_self_loop(x)
        g_self_loop = torch.sigmoid(self.W_gate_self_loop(x))

        return torch.relu(aggr_out + g_self_loop * m_self_loop)

    def get_latest_source_embeddings(self):
        return self.latest_source_embeddings

    def get_latest_target_embeddings(self):
        return self.latest_target_embeddings

    def get_latest_messages(self):
        return self.latest_messages

    def set_device(self, device):
        self.device = device


class SrlGcn(AbstractGNN):

    def __init__(self, dim, n_layers, n_relations):
        super(SrlGcn, self).__init__()

        self.dim = dim
        self.n_layers = n_layers
        self.n_relations = n_relations

        self.define_weights_and_layers()

    def is_adj_mat(self):
        return False

    def define_weights_and_layers(self):
        gnn_layers = []
        for layer in range(self.n_layers):
            gnn_layers.append(SrlGcnMessagePasser(self.dim,
                                                  self.dim,
                                                  n_relations=self.n_relations))

        gnn_layers = torch.nn.ModuleList(gnn_layers)
        self.gnn_layers = gnn_layers

    def get_initial_layer_input(self, vertex_embeddings):
        return vertex_embeddings

    def process_layer(self, vertex_embeddings=None,
                      edges=None,
                      edge_types=None,
                      edge_direction_cutoff=None,
                      gnn_layer=None,
                      message_scale=None,
                      message_replacement=None):

        if edge_direction_cutoff is None:
            print("Error: No cutoff for edge direction supplied")
            exit()

        if edge_types is None:
            print("Error: No edge types supplied")
            exit()

        layer_output = gnn_layer(vertex_embeddings,
                                 edge_direction_cutoff,
                                 edge_types,
                                 edges,
                                 message_scale=message_scale,
                                 message_replacement=message_replacement)

        return layer_output
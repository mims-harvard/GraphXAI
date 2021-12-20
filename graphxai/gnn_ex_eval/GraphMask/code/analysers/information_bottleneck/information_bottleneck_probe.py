import torch
from torch.nn import ReLU, Linear, Sigmoid

from code.abstract.abstract_torch_module import AbstractTorchModule
from code.utils.torch_utils.multiple_inputs_layernorm_linear import MultipleInputsLayernormLinear
from code.utils.torch_utils.squeezer import Squeezer


class InformationBottleneckProbe(AbstractTorchModule):

    device = None

    def __init__(self, vertex_embedding_dims, message_dims, h_dims, mean_and_var):
        AbstractTorchModule.__init__(self)

        gates = []
        for v_dim, m_dim, h_dim in zip(vertex_embedding_dims, message_dims, h_dims):
            gate_input_shape = [v_dim, m_dim, v_dim]
            gate = torch.nn.Sequential(
                MultipleInputsLayernormLinear(gate_input_shape, h_dim),
                ReLU(),
                Linear(h_dim, 1),
                Squeezer(),
                Sigmoid()
            )

            gates.append(gate)

        gates = torch.nn.ModuleList(gates)
        self.gates = gates

        self.q_z_loc = torch.FloatTensor(mean_and_var[0])
        self.q_z_scale = torch.FloatTensor(mean_and_var[1])

    def forward(self, gnn):
        latest_source_embeddings = gnn.get_latest_source_embeddings()
        latest_messages = gnn.get_latest_messages()
        latest_target_embeddings = gnn.get_latest_target_embeddings()

        all_gates = []
        all_baselines = []

        total_penalty = 0
        for i in range(len(self.gates)):
            gate_input = [latest_source_embeddings[i], latest_messages[i], latest_target_embeddings[i]]

            gates = self.gates[i](gate_input)

            p_z_r = torch.distributions.Normal(
                loc=gates.unsqueeze(-1) * gate_input[1]
                    + (1 - gates).unsqueeze(-1) * self.q_z_loc,
                scale=(self.q_z_scale + 1e-8) * (1 - gates).unsqueeze(-1),
            )

            q_z = torch.distributions.Normal(loc=self.q_z_loc, scale=(self.q_z_scale + 1e-8), )

            penalty = torch.distributions.kl_divergence(p_z_r, q_z).mean()

            all_gates.append(gates)
            all_baselines.append(p_z_r.rsample())
            total_penalty += penalty

        return all_gates, all_baselines, total_penalty

    def set_device(self, device):
        self.to(device)
import math
import sys

import torch
from torch.nn import ReLU, Linear

from code.abstract.abstract_torch_module import AbstractTorchModule
from code.utils.torch_utils.hard_concrete import HardConcrete
from code.utils.torch_utils.multiple_inputs_layernorm_linear import MultipleInputsLayernormLinear
from code.utils.torch_utils.squeezer import Squeezer
import ipdb

class GraphMaskProbe(AbstractTorchModule):

    device = None

    def __init__(self, vertex_embedding_dims, message_dims, h_dims):
        AbstractTorchModule.__init__(self)

        gates = []
        baselines = []
        for v_dim, m_dim, h_dim in zip(vertex_embedding_dims, message_dims, h_dims):
            import ipdb
            ipdb.set_trace()
            gate_input_shape = [v_dim, m_dim, v_dim]
            gate = torch.nn.Sequential(
                MultipleInputsLayernormLinear(gate_input_shape, h_dim),
                ReLU(),
                Linear(h_dim, 1),
                Squeezer(),
                HardConcrete()
            )

            gates.append(gate)

            baseline = torch.FloatTensor(m_dim)
            stdv = 1. / math.sqrt(m_dim)
            baseline.uniform_(-stdv, stdv)
            baseline = torch.nn.Parameter(baseline, requires_grad=True)

            baselines.append(baseline)

        gates = torch.nn.ModuleList(gates)

        self.gates = gates

        baselines = torch.nn.ParameterList(baselines)
        self.baselines = baselines

        # Initially we cannot update any parameters. They should be enabled layerwise
        for parameter in self.parameters():
            parameter.requires_grad = False

    def enable_layer(self, layer):
        print("Enabling layer "+str(layer), file=sys.stderr)
        for parameter in self.gates[layer].parameters():
            parameter.requires_grad = True

        self.baselines[layer].requires_grad = True


    def forward(self, gnn):
        latest_source_embeddings = gnn.get_latest_source_embeddings()
        latest_messages = gnn.get_latest_messages()
        latest_target_embeddings = gnn.get_latest_target_embeddings()

        gates = []
        total_penalty = 0
        for i in range(len(self.gates)):
            gate_input = [latest_source_embeddings[i], latest_messages[i], latest_target_embeddings[i]]
            gate, penalty = self.gates[i](gate_input)

            gates.append(gate)
            total_penalty += penalty

        return gates, self.baselines, total_penalty

    def set_device(self, device):
        self.to(device)
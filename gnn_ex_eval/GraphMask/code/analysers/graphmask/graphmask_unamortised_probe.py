import math

import torch

from code.abstract.abstract_torch_module import AbstractTorchModule
from code.utils.torch_utils.hard_concrete import HardConcrete


class GraphMaskUnamortisedProbe(AbstractTorchModule):


    def __init__(self, num_edges, num_layers, m_dim):
        AbstractTorchModule.__init__(self)

        self.hard_concrete = HardConcrete()

        gates = []
        baselines = []
        for layer in range(num_layers):
            gate_input = torch.FloatTensor(num_edges)
            gate_input.normal_(-1, 1)
            gate_input = torch.nn.Parameter(gate_input, requires_grad=True)

            gates.append(gate_input)

            baseline = torch.FloatTensor(m_dim)
            stdv = 1. / math.sqrt(m_dim)
            baseline.uniform_(-stdv, stdv)
            baseline = torch.nn.Parameter(baseline, requires_grad=True)

            baselines.append(baseline)

        gates = torch.nn.ParameterList(gates)
        self.gates = gates

        baselines = torch.nn.ParameterList(baselines)
        self.baselines = baselines

        # Initially we cannot update any parameters. They should be enabled layerwise
        for parameter in self.parameters():
            parameter.requires_grad = False

    def enable_layer(self, layer):
        self.gates[layer].requires_grad = True
        self.baselines[layer].requires_grad = True

    def forward(self):
        gates = []
        total_penalty = 0
        for i in range(len(self.gates)):
            gate_input = self.gates[i]
            gate, penalty = self.hard_concrete(gate_input)

            gates.append(gate)
            total_penalty += penalty

        return gates, self.baselines, total_penalty

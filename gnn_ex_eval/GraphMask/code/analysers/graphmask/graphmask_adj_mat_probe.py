import math
import sys

import torch
from torch.nn import ReLU, Linear

from code.abstract.abstract_torch_module import AbstractTorchModule
from code.utils.torch_utils.hard_concrete import HardConcrete
from code.utils.torch_utils.multiple_inputs_layernorm_linear import MultipleInputsLayernormLinear
from code.utils.torch_utils.squeezer import Squeezer


class GraphMaskAdjMatProbe(AbstractTorchModule):

    device = None

    def __init__(self, vertex_embedding_dims, message_dims, n_relations, h_dims):
        AbstractTorchModule.__init__(self)

        self.n_relations = n_relations

        self.hard_gates = HardConcrete()

        transforms = []
        baselines = []
        for v_dim, m_dim, h_dim in zip(vertex_embedding_dims, message_dims, h_dims):
            transform_src = torch.nn.Sequential(
                Linear(v_dim, h_dim),
                ReLU(),
                Linear(h_dim, m_dim * n_relations),
            )

            transforms.append(transform_src)

            baseline = torch.FloatTensor(m_dim)
            stdv = 1. / math.sqrt(m_dim)
            baseline.uniform_(-stdv, stdv)
            baseline = torch.nn.Parameter(baseline, requires_grad=True)

            baselines.append(baseline)

        transforms = torch.nn.ModuleList(transforms)
        self.transforms = transforms

        baselines = torch.nn.ParameterList(baselines)
        self.baselines = baselines

        # Initially we cannot update any parameters. They should be enabled layerwise
        for parameter in self.parameters():
            parameter.requires_grad = False

    def enable_layer(self, layer):
        print("Enabling layer "+str(layer), file=sys.stderr)
        for parameter in self.transforms[layer].parameters():
            parameter.requires_grad = True

        self.baselines[layer].requires_grad = True

    def forward(self, gnn):
        latest_vertex_embeddings = gnn.get_latest_vertex_embeddings()
        adj_mat = gnn.get_latest_adj_mat()

        gates = []
        total_penalty = 0
        for i in range(len(self.transforms)):
            srcs = latest_vertex_embeddings[i]

            src_shape = srcs.size()

            new_shape = list(src_shape)[:-1] + [self.n_relations, src_shape[-1]]
            transformed_src = self.transforms[i](srcs).view(new_shape).unsqueeze(1)

            tgt = srcs.unsqueeze(2).unsqueeze(2)

            a = (transformed_src * tgt).transpose(-2, 1)

            squeezed_a = a.sum(dim=-1)
            gate, penalty = self.hard_gates(squeezed_a, summarize_penalty=False)

            gate = gate * (adj_mat > 0).float()
            penalty_norm = (adj_mat > 0).sum().float()
            penalty = (penalty * (adj_mat > 0).float()).sum() / (penalty_norm + 1e-8)

            gates.append(gate)
            total_penalty += penalty

        return gates, self.baselines, total_penalty

    def set_device(self, device):
        self.to(device)
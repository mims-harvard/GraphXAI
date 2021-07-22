import torch

from code.abstract.abstract_torch_module import AbstractTorchModule


class IntegratedGradientsProbe(AbstractTorchModule):

    def __init__(self, num_edges, num_layers):
        AbstractTorchModule.__init__(self)
        pseudo_gates = torch.ones((num_layers, num_edges))
        self.pseudo_gates = torch.nn.Parameter(pseudo_gates, requires_grad=True)

    def forward(self):
        return self.pseudo_gates

import math
import torch
from torch.nn import Linear
from torch.nn import init


class XavierLinear(torch.nn.Module):

    """
    Linear transform initialized with Xavier uniform
    """

    def __init__(self, input_dim, output_dim, bias=True):
        super(XavierLinear, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.W = Linear(input_dim, output_dim, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(2.0 / float(self.input_dim + self.output_dim))
        a = math.sqrt(3.0) * std # Calculate uniform bounds from standard deviation

        init._no_grad_uniform_(self.W.weight, -a, a)

        if self.W.bias is not None:
            init._no_grad_zero_(self.W.bias)

    def forward(self, x):
        return self.W(x)

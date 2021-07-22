import math
import time
import os
import ipdb

import networkx as nx
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class IntegratedGrad:
    def __init__(self, model, label, x, edge_index, mapping, node_idx, subset, device):
        self.model = model
        self.label = label
        self.x = x
        self.edge_index = edge_index
        self.mapping = mapping
        self.node_idx = node_idx
        self.subset = subset
        self.device = device

    def __set_masks__(self):
        pass

    def explain(self, x):
        """Explain a single node prediction
        """
        steps = 40
        # scaled_inputs = tuple(baseline + (float(i) / steps) * (x - baseline) for i in range(0, steps + 1))

        self.model.eval()
        grads = []
        for i in range(0, steps + 1):
            with torch.no_grad():
                baseline = torch.zeros_like(x.clone())
                temp_x = baseline + (float(i) / steps) * (x.clone() - baseline)
            temp_x.requires_grad = True

            # for samp in scaled_inputs:
            output = self.model(temp_x.to(self.device), self.edge_index.to(self.device))

            # NLL_loss
            loss = F.nll_loss(output[self.mapping], self.label[self.mapping].to(self.device))
            loss.backward()
            grads.append(temp_x.grad[torch.where(self.subset == self.node_idx)[0].item(), :].numpy())

        # # Use trapezoidal rule to approximate the integral.
        # # See Section 4 of the following paper for an accuracy comparison between
        # # left, right, and trapezoidal IG approximations:
        # # "Computing Linear Restrictions of Neural Networks", Matthew Sotoudeh, Aditya V. Thakur
        # # https://arxiv.org/abs/1908.06214

        grads = np.array(grads)
        grads = (grads[:-1] + grads[1:]) / 2.0
        avg_grads = torch.mean(torch.from_numpy(grads), dim=0)
        integrated_gradients = ((x[torch.where(self.subset == self.node_idx)[0].item()] - baseline[0]) * avg_grads)
        return integrated_gradients

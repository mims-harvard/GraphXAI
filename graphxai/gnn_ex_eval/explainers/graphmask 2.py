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


class GraphMask:
    def __init__(self, model, x, node_idx, edge_index, hop, rho, device):
        self.model = model
        self.x = x
        self.node_idx = node_idx
        self.edge_index = edge_index
        self.hop = hop
        self.rho = rho
        self.device = device

    def explain(self, x):
        """Explain a single node prediction
		"""
        self.model = self.model.to('cpu')
        explainer = GraphLIME(self.model, hop=self.hop, rho=self.rho)
        explanation = explainer.explain_node(self.node_idx, x, self.edge_index)
        return torch.from_numpy(explanation)  # .to(self.device)

    def fit(self):
        print('Fitting the training data to train the parameters for gates and baselines')

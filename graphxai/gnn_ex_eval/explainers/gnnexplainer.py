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
from torch_geometric.nn import MessagePassing


class GNNExplainer:
    def __init__(self, model, label, x, edge_index, mapping, node_idx, subset, sub_num_nodes, sub_num_edges,
                 num_features, device):
        self.model = model
        self.label = label
        self.x = x
        self.edge_index = edge_index
        self.mapping = mapping
        self.node_idx = node_idx
        self.subset = subset
        self.sub_num_nodes = sub_num_nodes
        self.sub_num_edges = sub_num_edges
        self.num_features = num_features
        self.device = device

    def __set_masks__(self):
        # this is an initialization trick that increases optimization performance
        std = torch.nn.init.calculate_gain('relu') * np.sqrt(2.0 / (2 * self.sub_num_nodes))
        
        self.edge_mask = torch.nn.Parameter(torch.randn(self.sub_num_edges) * std)
        self.node_feat_mask = torch.nn.Parameter(torch.randn(self.num_features) * 0.1)

        # To tell PyG that a model is being used for explanation
        # we set the following on each layer in the model
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                # And we add the edge_mask to each layer's module object
                module.__edge_mask__ = self.edge_mask

    def explain_loss(self, logit, edge_mask, node_feat_mask, coeff):
        loss = -logit

        # take the sigmoid of the mask to get in [0,1]
        m = edge_mask.sigmoid()

        # add the sum of the mask elements to the loss
        loss = loss + coeff['edge_size'] * torch.sum(m)

        # calculate the element-wise entropy of the mask values
        # (low entropy implies a discrete variable - close to 1 or 0)
        entropy = -m * torch.log(m + 1e-15) - (1 - m) * torch.log(1 - m + 1e-15)
        loss = loss + coeff['edge_entropy_term'] * entropy.mean()

        # m = node_feat_mask.sigmoid()
        # loss = loss + coeff['feat_size'] * torch.sum(m)
        # entropy = -m * torch.log(m + 1e-15) - (1 - m) * torch.log(1 - m + 1e-15)
        # loss = loss + coeff['feat_entropy_term'] * entropy.mean()
        return loss

    def explain(self, x, edge_index):
        """Explain a single node prediction
        """
        coeff = {'edge_entropy_term': 1.0,
                 'feat_entropy_term': 0.1,
                 'edge_size': 0.005,
                 'feat_size': 1.0}

        self.model.eval()
        num_epochs = 200
        optimizer = torch.optim.Adam([self.edge_mask], lr=0.01)

        # train
        for epoch in range(1, num_epochs + 1):
            optimizer.zero_grad()

            # mask the feature matrix
            # h = sub_x * node_feat_mask.view(1, -1).sigmoid()

            # calculate the logit value by forward-pass
            logits = self.model(x.to(self.device), edge_index.to(self.device))
            logit = logits[self.mapping][0][self.label[self.mapping].item()]  # logits[self.mapping]

            # calculate regularized loss
            loss = self.explain_loss(logit, self.edge_mask, self.node_feat_mask, coeff)
            loss.backward()
            optimizer.step()

        # clear from the model
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None

        return self.edge_mask

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .root_explainer import RootExplainer


class Demo(RootExplainer):
    def __init__(self, model):
        super().__init__(model)

    def get_explanation_node(self, x):
        """
        Explain a node prediction
        """
        self.model.eval()
        output = self.model(x.to(self.device), self.edge_index.to(self.device))

        # NLL_loss
        loss = F.nll_loss(output[self.mapping], self.label[self.mapping].to(self.device))
        loss.backward()

        return x.grad[torch.where(self.subset==self.node_idx)[0].item(), :]

    def get_explanation_graph(self):
        """
        Explain a graph prediction
        """
        pass

    def get_explanation_link(self):
        """
        Explain an edge link prediction
        """
        pass

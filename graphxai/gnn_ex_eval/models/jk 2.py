import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, JumpingKnowledge
from torch.nn.utils import spectral_norm

class JK(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout): 
        super(JK, self).__init__()
        self.conv1 = spectral_norm(GCNConv(nfeat, nhid))
        self.convx= spectral_norm(GCNConv(nhid, nhid))
        self.jk = JumpingKnowledge(mode='max')
        self.final = nn.Linear(nhid, nclass)
        self.transition = nn.Sequential(
            nn.ReLU(),
        )

        for m in self.modules():
            self.weights_init(m)


    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index): 
        xs = []
        x = self.conv1(x, edge_index)
        x = self.transition(x)
        xs.append(x)
        for _ in range(1): 
            x = self.convx(x, edge_index)
            x = self.transition(x)
            xs.append(x)
        x = self.jk(xs)
        x = self.final(x)
        return x

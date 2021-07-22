import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn.utils import spectral_norm

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.body = GCN_Body(nfeat,nhid,dropout,nclass)
        self.fc = nn.Linear(nhid, ncla-1)
        self.softmax = nn.Softmax(dim=1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        x = self.body(x, edge_index)
        x = self.fc(x)
        return self.softmax(x)

# def GCN(nn.Module):
class GCN_Body(nn.Module):
    def __init__(self, nfeat, nhid, dropout, nclass):
        super(GCN_Body, self).__init__()
        self.gc1 = GCNConv(nfeat, nhid)
        self.gc2 = GCNConv(nhid, nhid)
        self.transition = nn.Sequential(
                    nn.Softplus(),
                    nn.Dropout(p=dropout)
                )

    def forward(self, x, edge_index):
        x = self.gc1(x, edge_index)
        x = self.transition(x)
        x = self.gc2(x, edge_index)
        return x

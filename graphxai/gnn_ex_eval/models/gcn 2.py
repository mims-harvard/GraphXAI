import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.nn.utils import spectral_norm


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GCNConv(nfeat, nhid, add_self_loops=False, normalize=False)  # body = GCN_Body(nfeat,nhid,dropout,nclass)
        self.transition = nn.Sequential(nn.ReLU(), nn.Dropout(p=dropout))
        self.gc2 = GCNConv(nhid, nhid, add_self_loops=False, normalize=False)
        self.fc = nn.Linear(nhid, nclass)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        x = self.transition(self.gc1(x, edge_index))
        x = self.gc2(x, edge_index)
        return F.softmax(self.fc(x), dim=-1)


class GCN_Body(nn.Module):
    def __init__(self, nfeat, nhid, dropout, nclass):
        super(GCN_Body, self).__init__()
        self.gc1 = GCNConv(nfeat, nhid)
        # self.gc1.aggr = 'mean'
        #self.gc2 = GCNConv(nhid, nhid)
        #self.gc2.aggr = 'mean'
        self.transition = nn.Sequential(
                    nn.ReLU(),
                    # nn.BatchNorm1d(nhid),
                    # nn.Dropout(p=dropout)
                )

    def forward(self, x, edge_index):
        x = self.gc1(x, edge_index)
        x = self.transition(x)
        # x = self.gc2(x, edge_index)
        return x


class GCN_graph(nn.Module):
    '''
    GCN model to perform graph-wide predictions
        - Uses global mean pooling for pooling layer'''
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GCNConv(nfeat, nhid, add_self_loops=False, normalize=False)  # body = GCN_Body(nfeat,nhid,dropout,nclass)
        self.transition = nn.Sequential(nn.ReLU(), nn.Dropout(p=dropout))
        self.gc2 = GCNConv(nhid, nhid, add_self_loops=False, normalize=False)
        self.fc = nn.Linear(nhid, nclass)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, batch):
        x = self.transition(self.gc1(x, edge_index))
        x = self.gc2(x, edge_index)
        x = global_mean_pool(x, batch)
        return F.softmax(self.fc(x), dim=-1)
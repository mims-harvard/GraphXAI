import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, DeepGraphInfomax
from torch.nn.utils import spectral_norm


class Encoder_DGI(nn.Module):
    def __init__(self, nfeat, nhid): 
        super(Encoder_DGI, self).__init__()
        self.hidden_ch = nhid
        self.conv = spectral_norm(GCNConv(nfeat, self.hidden_ch))
        self.activation = nn.PReLU()
        
    def corruption(self, x, edge_index): 
        # corrupted features are obtained by row-wise shuffling of the original features 
        # corrupted graph consists of the same nodes but located in different places 
        return x[torch.randperm(x.size(0))], edge_index
        
    def summary(self, z, *args, **kwargs): 
        return torch.sigmoid(z.mean(dim=0))

    def forward(self, x, edge_index): 
        x = self.conv(x, edge_index)
        x = self.activation(x)
        return x 


class Encoder_CLS(nn.Module):
    def __init__(self, nhid, nclass): 
        super(Encoder_CLS, self).__init__()
        self.conv = spectral_norm(GCNConv(nhid, nclass))

    def forward(self, x, edge_index): 
        return self.conv(x, edge_index)


class GraphInfoMax(nn.Module):
    def __init__(self, enc_dgi, enc_cls):
        super(GraphInfoMax, self).__init__()
        self.dgi_model = DeepGraphInfomax(enc_dgi.hidden_ch, enc_dgi, enc_dgi.summary, enc_dgi.corruption)
        self.cls_model = enc_cls

    def forward(self, x, edge_index): 
        pos_z, neg_z, summary = self.dgi_model(x, edge_index)
        output = self.cls_model(pos_z, edge_index)
        return output

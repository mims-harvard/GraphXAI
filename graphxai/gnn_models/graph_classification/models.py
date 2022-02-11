import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, Sequential, BatchNorm, JumpingKnowledge
from torch_geometric.nn import SAGEConv, GATConv

class SAGE_3layer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SAGE_3layer, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()

        x = global_mean_pool(x, batch)

        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x

class GAT_3layer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GAT_3layer, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()

        x = global_mean_pool(x, batch)

        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x

class JKNet_3layer(torch.nn.Module):
    def __init__(self, hidden_channels, in_channels, out_channels):
        super(JKNet_3layer, self).__init__()
        self.jknet = Sequential( 'x, edge_index', [
            (SAGEConv(in_channels, hidden_channels), 'x, edge_index -> x1'),
            (BatchNorm(hidden_channels), 'x1 -> x1'),
            (torch.nn.PReLU(), 'x1 -> x1'),
            (SAGEConv(hidden_channels, hidden_channels), 'x1, edge_index -> x2'),
            (BatchNorm(hidden_channels), 'x2 -> x2'),
            (torch.nn.PReLU(), 'x2 -> x2'),
            (SAGEConv(hidden_channels, hidden_channels), 'x2, edge_index -> x3'),
            (BatchNorm(hidden_channels), 'x3 -> x3'),
            (torch.nn.PReLU(), 'x3 -> x3'),
            (lambda x1, x2, x3: [x1, x2, x3], 'x1, x2, x3 -> xs'),
            (JumpingKnowledge('cat', hidden_channels, num_layers = 2), 'xs -> x'),
            (torch.nn.Linear(3 * hidden_channels, hidden_channels), 'x -> x'),
        ]
        )

        self.final_lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.jknet(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.final_lin(x)
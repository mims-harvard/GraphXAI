import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv, GINConv, BatchNorm

from graphxai.datasets.ba_houses_with_synth import BAHouses

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
#from graphxai.gnn_models.node_classification import train, test

class GCN_1layer(torch.nn.Module):
    def __init__(self, hidden_channels, input_feat, classes):
        super(GCN_1layer, self).__init__()
        self.conv1 = GCNConv(input_feat, hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return x

class GCN_2layer(torch.nn.Module):
    def __init__(self, hidden_channels, input_feat, classes):
        super(GCN_2layer, self).__init__()
        self.conv1 = GCNConv(input_feat, hidden_channels)
        self.batchnorm1 = BatchNorm(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.batchnorm1(x)
        x = torch.nn.PReLU()(x)
        x = self.conv2(x, edge_index)
        return x

class GCN_3layer(torch.nn.Module):
    def __init__(self, hidden_channels, input_feat, classes):
        super(GCN_3layer, self).__init__()
        self.gcn1 = GCNConv(input_feat, hidden_channels)
        self.batchnorm1 = BatchNorm(hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, hidden_channels)
        self.batchnorm2 = BatchNorm(hidden_channels)
        self.gcn3 = GCNConv(hidden_channels, classes)

    def forward(self, x, edge_index):
        x = self.gcn1(x, edge_index)
        x = self.batchnorm1(x)
        x = x.relu()
        x = self.gcn2(x, edge_index)
        x = self.batchnorm2(x)
        x = x.relu()
        x = self.gcn3(x, edge_index)
        return x

class GIN_2layer(torch.nn.Module):
    def __init__(self, hidden_channels, input_feat, classes):
        super(GIN_2layer, self).__init__()
        self.mlp_gin1 = torch.nn.Linear(input_feat, hidden_channels)
        self.gin1 = GINConv(self.mlp_gin1)
        self.batchnorm1 = BatchNorm(hidden_channels)
        self.mlp_gin2 = torch.nn.Linear(hidden_channels, classes)
        self.gin2 = GINConv(self.mlp_gin2)

    def forward(self, x, edge_index):
        x = self.gin1(x, edge_index)
        x = self.batchnorm1(x)
        x = torch.nn.PReLU()(x)
        x = self.gin2(x, edge_index)
        return x

class GIN_3layer(torch.nn.Module):
    def __init__(self, hidden_channels, input_feat, classes):
        super(GIN_3layer, self).__init__()
        self.mlp_gin1 = torch.nn.Linear(input_feat, hidden_channels)
        self.gin1 = GINConv(self.mlp_gin1)
        self.batchnorm1 = BatchNorm(hidden_channels)
        self.mlp_gin2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.gin2 = GINConv(self.mlp_gin2)
        self.batchnorm2 = BatchNorm(hidden_channels)
        self.mlp_gin3 = torch.nn.Linear(hidden_channels, classes)
        self.gin3 = GINConv(self.mlp_gin3)

    def forward(self, x, edge_index):
        x = self.gin1(x, edge_index)
        x = self.batchnorm1(x)
        x = x.relu()
        x = self.gin2(x, edge_index)
        x = self.batchnorm2(x)
        x = x.relu()
        x = self.gin3(x, edge_index)
        return x

def train(model, optimizer,
          criterion, data):
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(data.x, data.edge_index)  # Perform a single forward pass.
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss
    
def test(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    #print(data.y[data.test_mask].unique())
    test_score = f1_score(data.y[data.test_mask].tolist(), pred[data.test_mask].tolist())
    acc = accuracy_score(data.y[data.test_mask].tolist(), pred[data.test_mask].tolist())
    precision = precision_score(data.y[data.test_mask].tolist(), pred[data.test_mask].tolist())
    recall = recall_score(data.y[data.test_mask].tolist(), pred[data.test_mask].tolist())
    # print('Num predicted 0', (pred[data.test_mask] == 0).nonzero(as_tuple=True)[0].shape[0])
    # print('Num predicted 1', (pred[data.test_mask] == 1).nonzero(as_tuple=True)[0].shape[0])
    return test_score, acc, precision, recall

def labeling_rule_DEG(self):
    '''
    Labeling rule for each node
        - Rule: if (number of houses in k-hop == 1) 
            and (Degree > 1)
    '''
    
    def get_label(node_idx):
        # Count number of houses in k-hop neighborhood
        khop_edges = nx.bfs_edges(self.G, node_idx, depth_limit = self.num_hops)
        nodes_in_khop = set(np.unique(list(khop_edges))) - set([node_idx])
        num_unique_houses = len(np.unique([self.G.nodes[ni]['shape'] \
                for ni in nodes_in_khop if self.G.nodes[ni]['shape'] > 0 ]))
        # Enfore logical condition:
        return int(num_unique_houses == 1 and self.G.nodes[node_idx]['x'][0] > 1)
            
    return get_label

def labeling_rule_CC(self):
    '''
    Labeling rule for each node
        - Rule: if (number of houses in k-hop > 1)
            and (CC > Average CC of all nodes)
    '''

    avg_ccs = np.mean([self.G.nodes[i]['x'][1] for i in self.G.nodes])
    
    def get_label(node_idx):
        # Count number of houses in k-hop neighborhood
        khop_edges = nx.bfs_edges(self.G, node_idx, depth_limit = self.num_hops)
        nodes_in_khop = set(np.unique(list(khop_edges))) - set([node_idx])
        num_unique_houses = len(np.unique([self.G.nodes[ni]['shape'] \
                for ni in nodes_in_khop if self.G.nodes[ni]['shape'] > 0 ]))
        # Enfore logical condition:
        return int(num_unique_houses > 1 and self.G.nodes[node_idx]['x'][1] < avg_ccs)
            
    return get_label

def labeling_only_graph(self):
    def get_label(node_idx):
        khop_edges = nx.bfs_edges(self.G, node_idx, depth_limit = self.num_hops)
        nodes_in_khop = set(np.unique(list(khop_edges))) - set([node_idx])
        num_unique_houses = len(np.unique([self.G.nodes[ni]['shape'] \
                for ni in nodes_in_khop if self.G.nodes[ni]['shape'] > 0 ]))
        return int(num_unique_houses == 1)

    return get_label

def labeling_only_features(self):
    def get_label(node_idx):
        return int(self.G.nodes[node_idx]['x'][0] > 1)
    return get_label


if __name__ == '__main__':
    bah = BAHouses(
    num_hops=2,
    n=30,
    m=1,
    num_houses=1,
    plant_method='local',
    label_rule = labeling_only_features)


    ylist = bah.graph.y.tolist()
    node_colors = [ylist[i] for i in bah.G.nodes]
    pos = nx.kamada_kawai_layout(bah.G)
    fig, ax = plt.subplots()
    nx.draw(bah.G, pos, node_color = node_colors, ax=ax)
    ax.set_title('Condition: if degree > 1 (Only Features)')
    plt.tight_layout()
    plt.show()
    exit()

    data = bah.get_graph(use_fixed_split=False, split_sizes = [0.7, 0.3, 0]) # Get torch_geometric data

    model = GCN_3layer(64, input_feat=3, classes=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    all_f1s = []
    for epoch in range(1,1000):
        loss = train(model, optimizer, criterion, data)
        f1, acc, prec, rec = test(model, data)
        all_f1s.append(f1)
        #print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test F1: {f1:.4f}, Test Acc: {acc:.4f}, P: {prec:.4f}, R: {rec:.4f}')

    print('Count 0:', (data.y == 0).nonzero(as_tuple=True)[0].shape[0])
    print('Count 1:', (data.y == 1).nonzero(as_tuple=True)[0].shape[0])
    print('Max', max(all_f1s))
    print('Epochs:', np.argmax(all_f1s))

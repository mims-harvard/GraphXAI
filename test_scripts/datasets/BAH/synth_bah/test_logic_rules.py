import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv, GINConv, BatchNorm

from graphxai.datasets.BA_shapes.ba_houses import BAHouses

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from graphxai.gnn_models.node_classification.testing import *

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
        '''
        '''
        # Count number of houses in k-hop neighborhood
        khop_edges = nx.bfs_edges(self.G, node_idx, depth_limit = self.num_hops)
        nodes_in_khop = set(np.unique(list(khop_edges))) - set([node_idx])
        num_unique_houses = len(np.unique([self.G.nodes[ni]['shape'] \
                for ni in nodes_in_khop if self.G.nodes[ni]['shape'] > 0 ]))
        # Enfore logical condition:
        return int(num_unique_houses > 1 and self.G.nodes[node_idx]['x'][1] < avg_ccs)
            
    return get_label

def labeling_only_graph(self):
    '''
    Labeling based on only the graph (number of houses in hop being equal to 1)
    '''
    def get_label(node_idx):
        khop_edges = nx.bfs_edges(self.G, node_idx, depth_limit = self.num_hops)
        nodes_in_khop = set(np.unique(list(khop_edges))) - set([node_idx])
        num_unique_houses = len(np.unique([self.G.nodes[ni]['shape'] \
                for ni in nodes_in_khop if self.G.nodes[ni]['shape'] > 0 ]))
        return int(num_unique_houses == 1)

    return get_label

def labeling_only_features(self):
    '''
    Labeling based on only the features (degree)
    '''
    def get_label(node_idx):
        return int(self.G.nodes[node_idx]['x'][0] > 1)
    return get_label


if __name__ == '__main__':
    bah = BAHouses(
    num_hops=2,
    n=1000,
    m=1,
    num_houses=1,
    shape_insert_strategy='local',
    # Change the label rule:
    label_rule = labeling_only_features)

    data = bah.get_graph(use_fixed_split=False, split_sizes = [0.7, 0.3, 0]) # Get torch_geometric data

    # Change the model architecture there:
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

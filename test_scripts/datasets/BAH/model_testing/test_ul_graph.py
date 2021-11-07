import time
import torch
import pandas as pd
import networkx as nx
import numpy as np 
import matplotlib.pyplot as plt
from torch_geometric.utils import from_networkx
from sklearn.model_selection import train_test_split

from graphxai.datasets.utils.bound_graph import build_bound_graph
from graphxai.gnn_models.node_classification.testing import *
from graphxai.datasets import BAShapes

class Hyperparameters:
    num_hops = 2
    n = 30 # Invalid for bound
    m = 1 # Invalid for bound
    num_shapes = 5 # Invalid
    shape_insert_strategy = 'bound_12'
    shape_upper_bound = 1
    shape = 'house'
    labeling_method = 'edge'
    feature_method = 'gaussian_lv'

hyp = Hyperparameters
args = {key:value for key, value in hyp.__dict__.items() if not key.startswith('__') and not callable(value)}
bah = BAShapes(**args)

# data = from_networkx(G)

# x = []
# for n in G.nodes:
#     x.append([G.degree(n), nx.clustering(G, nodes = n)])

# data.x = torch.tensor(x, dtype=torch.float32)
# data.y = torch.tensor(y, dtype=torch.long) - 1

n_trials = 10

max_f1s = []

for n in range(n_trials):

    # train_mask, test_mask = train_test_split(torch.tensor(range(data.x.shape[0])), 
    #     test_size = 0.2, stratify = data.y)
    # train_tensor, test_tensor = torch.zeros(data.y.shape[0], dtype=bool), torch.zeros(data.y.shape[0], dtype=bool)
    # train_tensor[train_mask] = 1
    # test_tensor[test_mask] = 1

    # data.train_mask = train_tensor
    # data.test_mask = test_tensor

    bah = BAShapes(**args)
    data = bah.get_graph(use_fixed_split=False, split_sizes = [0.7, 0.3, 0])

    #model = GIN_2layer(64, input_feat=10, classes=2)
    model = GCN_1layer(input_feat=10, classes=2)

    count_0 = (data.y == 0).nonzero(as_tuple=True)[0].shape[0]
    count_1 = (data.y == 1).nonzero(as_tuple=True)[0].shape[0]

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    all_f1s = []
    all_acs = []
    all_prec = []
    all_rec = []
    for epoch in range(1,400):
        loss = train(model, optimizer, criterion, data)
        #print('Loss', loss.item())
        f1, acc, prec, rec = test(model, data)
        all_f1s.append(f1)
        all_acs.append(acc)
        all_prec.append(prec)
        all_rec.append(rec)
        #print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test F1: {f1:.4f}, Test Acc: {acc:.4f}, P: {prec:.4f}, R: {rec:.4f}')

    max_f1s.append(max(all_f1s))

print('Count 0:', count_0)
print('Count 1:', count_1)
#print('Max', max(all_f1s))
print('Avg F1', np.mean(max_f1s))
print('Epochs:', np.argmax(all_f1s))

# x = list(range(len(all_f1s)))
# plt.plot(x, all_f1s, label = 'F1')
# plt.plot(x, all_acs, label = 'Accuracy')
# plt.plot(x, all_prec, label = 'Precision')
# plt.plot(x, all_rec, label = 'Recall')
# #plt.title('Metrics on {} ({} layers), {} Features'.format(sys.argv[2], sys.argv[3], sys.argv[1]))
# plt.xlabel('Epochs')
# plt.ylabel('Metric')
# plt.legend()
# plt.show()
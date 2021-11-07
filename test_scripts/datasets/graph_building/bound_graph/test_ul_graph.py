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

from graphxai.utils import khop_subgraph_nx

G = build_bound_graph(num_subgraphs = 10, num_hops=2, prob_connection = 0.9)

y = [d['shapes_in_khop'] for _, d in G.nodes(data=True)]
shape = [d['shape'] for _,d in G.nodes(data=True)]
print(np.unique(shape))
print(np.unique(y))

print(pd.Series(y).value_counts())
print('APL', nx.average_shortest_path_length(G))

pos = nx.kamada_kawai_layout(G)
#nx.draw(G, pos, node_color = shape)
#plt.colorbar()
#plt.show()

count_stop = 0
ns = list(range(G.number_of_nodes()))
ns_set = set(ns)
print('nodes', ns)
print('nodes')
# print(G.nodes)
# print([G.nodes[i] for i in range(G.number_of_nodes())])
for t in G.nodes:
    # print(t)
    # print(type(t))
    if t in ns:
        #nodes.remove(t)
        h = 1
    else:
        print('t not in nodes', t)
        print('Should be', ns[t])
    nodes = khop_subgraph_nx(node_idx = t, num_hops = 2, G = G)
    nodes_in_khop = list(set(nodes) - set([t])) 
    unique = np.unique([G.nodes[n]['shape_number'] for n in nodes_in_khop])
    #if 0 not in unique and count_stop < 5:
    #if G.nodes[t]['shapes_in_khop'] != (len(unique) - 1 if 0 in unique else len(unique)):
    if G.nodes[t]['shapes_in_khop'] == 2 and G.nodes[t]['shape'] == 1:
        count_stop += 1
        show_nodes = khop_subgraph_nx(node_idx = t, num_hops = 4, G = G)
        node_color = [G.nodes[n]['shape_number'] if n != t else 15 for n in show_nodes]
        labs = {n:G.nodes[n]['shapes_in_khop'] for n in show_nodes}
        subG = G.subgraph(show_nodes)
        pos = nx.spring_layout(subG)
        print('Node {} unique: {}'.format(t, unique))
        nx.draw(subG, pos, node_color = node_color)
        nx.draw_networkx_labels(subG, pos, labels = labs)
        plt.show()

    ns_set.remove(t)

print('Left over', ns_set)
print(G.nodes)
print(len(G.nodes))

exit()


# data = from_networkx(G)

# x = []
# for n in G.nodes:
#     x.append([G.degree(n), nx.clustering(G, nodes = n)])

# data.x = torch.tensor(x, dtype=torch.float32)
# data.y = torch.tensor(y, dtype=torch.long) - 1

# n_trials = 10

# max_f1s = []

# for n in range(n_trials):
#     train_mask, test_mask = train_test_split(torch.tensor(range(data.x.shape[0])), 
#         test_size = 0.2, stratify = data.y)
#     train_tensor, test_tensor = torch.zeros(data.y.shape[0], dtype=bool), torch.zeros(data.y.shape[0], dtype=bool)
#     train_tensor[train_mask] = 1
#     test_tensor[test_mask] = 1

#     data.train_mask = train_tensor
#     data.test_mask = test_tensor

#     model = GCN_3layer(64, input_feat=2, classes=2)

#     count_0 = (data.y == 0).nonzero(as_tuple=True)[0].shape[0]
#     count_1 = (data.y == 1).nonzero(as_tuple=True)[0].shape[0]

#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
#     criterion = torch.nn.CrossEntropyLoss()

#     all_f1s = []
#     all_acs = []
#     all_prec = []
#     all_rec = []
#     for epoch in range(1,400):
#         loss = train(model, optimizer, criterion, data)
#         #print('Loss', loss.item())
#         f1, acc, prec, rec = test(model, data)
#         all_f1s.append(f1)
#         all_acs.append(acc)
#         all_prec.append(prec)
#         all_rec.append(rec)
#         #print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test F1: {f1:.4f}, Test Acc: {acc:.4f}, P: {prec:.4f}, R: {rec:.4f}')

#     max_f1s.append(max(all_f1s))

# print('Count 0:', count_0)
# print('Count 1:', count_1)
# #print('Max', max(all_f1s))
# print('Avg F1', np.mean(max_f1s))
# print('Epochs:', np.argmax(all_f1s))

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
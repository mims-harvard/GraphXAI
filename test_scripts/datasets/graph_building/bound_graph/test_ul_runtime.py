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

# Time analysis:
#num_hops_search = [1]

#prob_connection_search = [0.02, 0.025, 0.03, 0.04]
prob_connection_search = [0.01, 0.02, 0.025, 0.03, 0.04, 0.05, 0.1, 0.15, 0.5, 0.75, 1]
num_subgraph_search = [5, 10, 20, 30, 40, 50, 75, 100]

time_dict = {subg: [] for subg in num_subgraph_search}
size_dict = {subg: [] for subg in num_subgraph_search}

for p in prob_connection_search:
    for sub in num_subgraph_search:
        print('prob connect:', p, 'Num_subg:', sub)
        start_time = time.time()
        G = build_bound_graph(num_subgraphs = sub, num_hops = 1, prob_connection = p)
        # nx.draw(G)
        # plt.show()
        size_graph = G.number_of_nodes()
        t = time.time() - start_time
        print('\t Time:', t)
        print('\t Size:', size_graph)
        print('\t Is Connected?:', nx.is_connected(G))
        time_dict[sub].append(t)
        size_dict[sub].append(size_graph)

pd.DataFrame(time_dict, index = prob_connection_search).to_csv('runtimes.csv')
pd.DataFrame(size_dict, index = prob_connection_search).to_csv('sizes.csv')
import sys
import torch
import networkx as nx
import matplotlib.pyplot as plt

from graphxai.datasets.shape_graph import ShapeGraph

def parse_args():
    arg_dict = {}

    assert len(sys.argv) == 4, 'usage: python3 test_one_ul.py <num_subgraphs> <prob_connection> <subgraph_size>'

    arg_dict['num_subgraphs'] = int(sys.argv[1])
    arg_dict['prob_connection'] = float(sys.argv[2])
    arg_dict['subgraph_size'] = int(sys.argv[3])

    return arg_dict


args = parse_args()

bah = ShapeGraph(**args, model_layers=3)
data = bah.get_graph()
G = bah.G

print('\t Size:', bah.num_nodes)
print('\t Class 0:', torch.sum(data.y == 0))
print('\t Class 1:', torch.sum(data.y == 1))

# Get degree distribution of G:
degrees = sorted([d for n, d in G.degree()])

plt.hist(degrees, color = 'green')
plt.title('Degree Distribution')
plt.show()

#c = [d['shape'] for n, d in G.nodes(data=True)]

# nx.draw(G, node_color = c)
# plt.show()

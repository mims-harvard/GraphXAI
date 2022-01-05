import sys, time, pickle
import torch
import networkx as nx
import matplotlib.pyplot as plt

from graphxai.datasets.shape_graph import ShapeGraph

def parse_args():
    arg_dict = {}

    assert len(sys.argv) >= 4, 'usage: python3 test_one_ul.py <num_subgraphs> <prob_connection> <subgraph_size>'

    arg_dict['num_subgraphs'] = int(sys.argv[1])
    arg_dict['prob_connection'] = float(sys.argv[2])
    arg_dict['subgraph_size'] = int(sys.argv[3])

    if len(sys.argv) > 4:
        arg_dict['variant'] = int(sys.argv[4])
    else:
        arg_dict['variant'] = 1

    return arg_dict


args = parse_args()

start_time = time.time()
bah = ShapeGraph(**args, model_layers=3, verify = False)
#bah.dump('large_graph.pickle')
#pickle.dump(bah, open('ShapeGraph_large_graph.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)
print('Time to make:', time.time() - start_time)
data = bah.get_graph()
G = bah.G

print('\t Size:', bah.num_nodes)
print('\t Class 0:', torch.sum(data.y == 0))
print('\t Class 1:', torch.sum(data.y == 1))

# Get degree distribution of G:
degrees = sorted([d for n, d in G.degree()])

variant_code = 'PA' if args['variant'] == 1 else 'Original'

plt.hist(degrees, color = 'green')
plt.title('Degree Distribution - {}'.format(variant_code))
plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.show()

#c = [d['shape'] for n, d in G.nodes(data=True)]

# nx.draw(G, node_color = c)
# plt.show()

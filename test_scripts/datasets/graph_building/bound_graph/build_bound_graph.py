import sys
import networkx as nx
import matplotlib.pyplot as plt
from graphxai.datasets.utils.bound_graph import build_bound_graph

def parse_argv():

    assert len(sys.argv) > 2, \
        'usage: python3 build_bound_graph.py <num_subgraphs> <prob_connection> <base_graph (optional)>'

    args = {}

    args['num_hops'] = 1
    args['num_subgraphs'] = int(sys.argv[1])
    #args['inter_sg_connections'] = int(sys.argv[2])
    args['prob_connection'] = float(sys.argv[2])
    args['base_graph'] = 'ba' if len(sys.argv) == 3 else sys.argv[3]

    return args

args = parse_argv()

G = build_bound_graph(**args)
# nx.draw(G)
# plt.show()

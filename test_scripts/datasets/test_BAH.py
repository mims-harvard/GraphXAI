import sys
import networkx as nx
import matplotlib.pyplot as plt

from graphxai.datasets.ba_houses_generators import generate_BAHouses_graph_local, generate_BAHouses_graph_global

if sys.argv[1] == 'local':
    G, _, y, _, = generate_BAHouses_graph_local(
        n=100, 
        m = 1, 
        k = 1, 
        num_hops = 2,
        get_data = False,
        in_hood_numbering=True)
elif sys.argv[1] == 'global':
    G, _, y, _, = generate_BAHouses_graph_global(
        n=50, 
        m = 1, 
        num_houses = 7, 
        num_hops = 2,
        get_data = False)

# Get nodes in houses, highlight by that:
#node_weights = [G.nodes[n]['house'] for n in G.nodes]
node_weights = y

pos = nx.kamada_kawai_layout(G)
nx.draw(G.to_undirected(), pos, node_color = node_weights)
plt.show()
import networkx as nx
import matplotlib.pyplot as plt

from graphxai.datasets.utils.bound_graph_pref_att import build_bound_graph

G = build_bound_graph(num_subgraphs = 3, prob_connection = 0.5, subgraph_size = 8, show_subgraphs = True)

c = [int(G.nodes[n]['shape'] == 0) for n in G.nodes]
pos = nx.kamada_kawai_layout(G)
nx.draw(G, node_color = c, cmap = 'brg')
plt.show()
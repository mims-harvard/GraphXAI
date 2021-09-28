import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

G = nx.barabasi_albert_graph(100, 1)
node_idx = random.choice(list(G.nodes))
edges = nx.bfs_edges(G, node_idx, depth_limit=1)
e = list(edges)
#print(list(edges))
nodes = np.unique(e)
print(nodes)

subG = nx.Graph()
subG.add_nodes_from(nodes)
subG.add_edges_from(e)

node_weights = [(1 if i == node_idx else 0) for i in subG.nodes]
nx.draw(subG, node_color = node_weights)
plt.show()

nx.draw(G, node_color = [1 if i == node_idx else 0 for i in G.nodes])
plt.show()
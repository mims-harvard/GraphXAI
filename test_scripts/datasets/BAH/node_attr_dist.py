import sys
import torch
import networkx as nx
import matplotlib.pyplot as plt

from graphxai.datasets.ba_houses_generators import generate_BAHouses_graph_local, generate_BAHouses_graph_global


G, x, y, _, = generate_BAHouses_graph_local(
    n=100, 
    m = 1, 
    k = 1, 
    num_hops = 2,
    get_data = False,
    in_hood_numbering=True,
    label_strategy = 1)

# Get nodes in houses, highlight by that:
#node_weights = [G.nodes[n]['house'] for n in G.nodes]
node_weights = y

ymask = torch.tensor(y, dtype=torch.bool)

fig, axs = plt.subplots(1, 3)

titles = ['degree', 'clustering coef', 'centrality']

for i in range(len(axs)):
    #axs[i].scatter(y, x[:,i].tolist())
    axs[i].violinplot([x[ymask.logical_not(),i].tolist(), x[ymask,i].tolist()], [0, 1])
    axs[i].set_title(titles[i])
    axs[i].set_ylabel(titles[i])
    axs[i].set_xlabel('Label')

plt.show()

pos = nx.kamada_kawai_layout(G)
nx.draw(G.to_undirected(), pos, node_color = node_weights)
plt.show()
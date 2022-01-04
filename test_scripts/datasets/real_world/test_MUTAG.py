import torch
import networkx as nx
import matplotlib.pyplot as plt

from torch_geometric.utils import to_networkx
from graphxai.datasets.real_world.MUTAG import MUTAG

mutag = MUTAG(root = '.')

g1 = mutag.graphs[109]

print(torch.sum(g1.x, dim = 0))

node_c = [g1.x[i,1].item() for i in range(g1.x.shape[0])]
print(node_c)

nxg1 = to_networkx(g1)
pos = nx.kamada_kawai_layout(nxg1)
nx.draw(nxg1, pos, node_color = node_c)
plt.show()

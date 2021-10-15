import networkx as nx
import matplotlib.pyplot as plt

from graphxai.datasets.ba_houses import BAHousesRandomOneHotFeatures

# def label_rule(self):
#     def get_label(node_idx):
#         return (self.G.nodes[node_idx]['shape'] > 0)
#     return get_label

bah = BAHousesRandomOneHotFeatures(
    num_hops=1,
    n=50,
    m=1,
    num_houses=None,
    shape_insert_strategy='neighborhood upper bound',
    shape_upper_bound = 1)

print(bah.shapes_in_graph)
print(bah.graph.y)
ylist = bah.graph.y.tolist()
node_colors = [ylist[i] for i in bah.G.nodes]

print('Number 0:', (bah.graph.y == 0).nonzero(as_tuple=True)[0].shape[0])
print('Number 1:', (bah.graph.y == 1).nonzero(as_tuple=True)[0].shape[0])

pos = nx.kamada_kawai_layout(bah.G)
fig, ax = plt.subplots()
nx.draw(bah.G, pos, node_color = node_colors, ax=ax)
ax.set_title('Labeling Rule: If num_houses_in_1hop == 1 and feature[1] == 1')
plt.tight_layout()
plt.show()
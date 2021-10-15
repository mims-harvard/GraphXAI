import networkx as nx
import matplotlib.pyplot as plt

from graphxai.datasets.ba_houses import BAHouses

def label_rule(self):
    def get_label(node_idx):
        return (self.G.nodes[node_idx]['shape'] > 0)
    return get_label

bah = BAHouses(
    num_hops=2,
    n=100,
    m=1,
    num_houses=10,
    shape_insert_strategy='neighborhood upper bound',
    shape_upper_bound = 1,
    labeling_rule=label_rule)

print(bah.shapes_in_graph)
print(bah.graph.y)
ylist = bah.graph.y.tolist()
node_colors = [ylist[i] for i in bah.G.nodes]

print('Number 0:', (bah.graph.y == 0).nonzero(as_tuple=True)[0].shape[0])
print('Number 1:', (bah.graph.y == 1).nonzero(as_tuple=True)[0].shape[0])

pos = nx.kamada_kawai_layout(bah.G)
fig, ax = plt.subplots()
nx.draw(bah.G, pos, node_color = node_colors, ax=ax)
ax.set_title('Neighborhood upper-bound (2-hop window)')
plt.tight_layout()
plt.show()
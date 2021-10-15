import networkx as nx
import matplotlib.pyplot as plt

from graphxai.datasets.ba_houses_with_synth import BAHouses

bah = BAHouses(
    num_hops=2,
    n=100,
    m=1,
    num_houses=1,
    shape_insert_strategy='local',
    shape = 'random')

print(bah.shapes_in_graph)
print(bah.graph.y)
ylist = bah.graph.y.tolist()
node_colors = [ylist[i] for i in bah.G.nodes]

pos = nx.kamada_kawai_layout(bah.G)
fig, ax = plt.subplots()
nx.draw(bah.G, pos, node_color = node_colors, ax=ax)
ax.set_title('Condition: if nhouses_in_2hop > 1 and CC < Avg(CC)')
plt.tight_layout()
plt.show()
import random
import networkx as nx
import matplotlib.pyplot as plt

from graphxai.utils import khop_subgraph_nx

house = nx.house_graph()

base_graph = nx.barabasi_albert_graph(n=10, m = 1)

# Choose pivot:
nx.set_node_attributes(house, 1, 'shape')
nx.set_node_attributes(base_graph, 0, 'shape')

to_pivot = random.choice(list(house.nodes))
pivot = random.choice(list(base_graph.nodes))


# ------------------------ Draw house -----------------------------
new_node = 6
connect = random.choice(list(house.nodes))
house.add_edge(new_node, connect)

house_color = [int(n == new_node) for n in house.nodes]

print(house_color)

nx.draw(house, node_color = house_color, cmap = 'brg')
plt.show()
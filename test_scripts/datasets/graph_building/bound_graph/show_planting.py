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
house_color = [int(n == to_pivot) for n in house.nodes]

print(house_color)

nx.draw(house, node_color = house_color, cmap = 'brg')
plt.show()

# ------------------------ Draw base -----------------------------
base_color = [int(n == pivot) for n in base_graph.nodes]
nx.draw(base_graph, node_color = base_color, cmap = 'brg')
plt.show()

# ------------------------ Merge graphs -----------------------------
convert = {to_pivot: pivot}

mx_nodes = max(list(base_graph.nodes))
i = 1
in_house = [pivot]
for n in house.nodes:
    if not (n == to_pivot):
        convert[n] = mx_nodes + i
        in_house.append(mx_nodes+i)
    i += 1

house = nx.relabel.relabel_nodes(house, convert)

base_graph.add_nodes_from(house.nodes(data=True))
base_graph.add_edges_from(house.edges)


# ------------------------ Draw whole graph -----------------------------
whole_colors = []
for n in base_graph.nodes:
    if n == pivot:
        val = 0.5
    elif n in in_house:
        val = 1
    else:
        val = 0

    whole_colors.append(val)

nx.draw(base_graph, node_color = whole_colors, cmap = 'brg')
plt.show()

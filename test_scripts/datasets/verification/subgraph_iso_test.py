import random
import networkx as nx
import matplotlib.pyplot as plt

iso = nx.house_graph()

# Plant multiple house graphs into a larger graph:
base = nx.barabasi_albert_graph(n = 30, m = 1)
nx.set_node_attributes(base, 0, 'shape')

floor_counter = base.number_of_nodes()

for i in range(1, 6):
    new_shape = nx.house_graph()
    nx.set_node_attributes(new_shape, i, 'shape')
    #Pick random nodes from each:
    relabeler = {ns: floor_counter + ns for ns in new_shape.nodes}
    new_shape = nx.relabel.relabel_nodes(new_shape, relabeler)
    floor_counter += new_shape.number_of_nodes()

    rand_house = random.choice(list(new_shape.nodes))
    rand_base = random.choice(list(base.nodes))

    base.add_nodes_from(new_shape.nodes(data=True))
    base.add_edges_from(new_shape.edges)
    base.add_edge(rand_house, rand_base)

# node_color = [d['shape'] for n, d in base.nodes(data=True)]
# nx.draw(base, node_color = node_color)
# plt.show()

# Subgraph match:
matcher = nx.algorithms.isomorphism.ISMAGS(graph = base, subgraph = iso)
i = 0
for isomorphism in matcher.find_isomorphisms():
    print(isomorphism.keys())
    i += 1
    print(i)

print(i)
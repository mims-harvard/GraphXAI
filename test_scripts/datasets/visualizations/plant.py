import networkx as nx
import matplotlib.pyplot as plt

BAG = nx.barabasi_albert_graph(n=10, m=1)

pos = nx.kamada_kawai_layout(BAG)

print(type(pos))
print(pos)

nx.draw(BAG, pos)
plt.show()
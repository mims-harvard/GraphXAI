import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 130
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.sans-serif'] = ['Lucida Console']

import networkx as nx
import matplotlib.pyplot as plt
plt.style.use('ggplot') # Use ggplot style

from graphxai.datasets.load_synthetic import load_ShapeGraph

SG = load_ShapeGraph(number = 1)

G = SG.G

degrees = sorted([d for n, d in G.degree()])

plt.hist(degrees, color = 'green')
plt.title('Degree Distribution - ShapeGraph')
plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()
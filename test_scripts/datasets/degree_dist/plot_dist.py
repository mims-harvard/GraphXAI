import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 100
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.sans-serif'] = ['Lucida Console']

import networkx as nx
import matplotlib.pyplot as plt
from read_edgeind import degdist, get_edge_index
#plt.style.use('ggplot') # Use ggplot style

from graphxai.datasets.load_synthetic import load_ShapeGGen

#SG = load_ShapeGGen(number = 1)
SG = load_ShapeGGen('data/ShapeGGen/new_unzipped/SG_homophilic.pickle', root = '/Users/owenqueen/Desktop/HMS_research/graphxai_project/GraphXAI')

G = SG.G

degrees = sorted([d for n, d in G.degree()])

plt.hist(degrees, color = 'green', bins = 15)
#plt.title('Degree Distribution - ShapeGGen')
plt.xlabel('Degree')
plt.ylabel('log( Frequency )')
fig = plt.gcf()
fig.set_size_inches(4, 3)
plt.tight_layout()
plt.yscale('log')
plt.xticks([i*2 for i in range(8)])
plt.savefig('sg.pdf', format='pdf')
plt.show()

# Visualization of Recidivism:
# rec_path = '/Users/owenqueen/Desktop/HMS_research/graphxai_project/GraphXAI/data/RW/bail/bail_edges.txt'
# degrees = degdist(get_edge_index(rec_path))
# plt.hist(degrees, color = 'red')
# #plt.title('De')
# plt.xlabel('Degree')
# plt.ylabel('log(Frequency)')
# plt.tight_layout()
# plt.xlim(0, 1000)
# plt.show()
#exit()

# Visualization of ER graph:
import networkx as nx
ERG = nx.erdos_renyi_graph(n=SG.num_nodes, p = 0.0005)
erdist = sorted([d for n, d in ERG.degree()])
plt.hist(erdist, color = 'red', bins = 15)
plt.xlabel('Degree')
plt.ylabel('log( Frequency )')
plt.xticks([i*4 for i in range(6)])
fig = plt.gcf()
fig.set_size_inches(4, 3)
plt.tight_layout()
plt.yscale('log')
plt.savefig('ER.pdf', format='pdf')
plt.show()
#exit()

# Visualization of German Credit
rec_path = '/Users/owenqueen/Desktop/HMS_research/graphxai_project/GraphXAI/data/RW/german/german_edges.txt'
degrees = degdist(get_edge_index(rec_path))
plt.hist(degrees, color = 'blue', bins = 15)
#plt.title('De')
plt.xlabel('Degree')
plt.ylabel('log( Frequency )')
fig = plt.gcf()
fig.set_size_inches(4, 3)
plt.tight_layout()
plt.yscale('log')
plt.savefig('german.pdf', format='pdf')
plt.show()
#exit()


# Visualization of Credit Defaulter:
rec_path = '/Users/owenqueen/Desktop/HMS_research/graphxai_project/GraphXAI/data/RW/credit/credit_edges.txt'
degrees = degdist(get_edge_index(rec_path))
plt.hist(degrees, color = 'purple', bins = 15)
#plt.title('De')
plt.xlabel('Degree')
plt.ylabel('log( Frequency )')
fig = plt.gcf()
fig.set_size_inches(4, 3)
plt.tight_layout()
plt.yscale('log')
plt.savefig('credit.pdf', format='pdf')
plt.show()

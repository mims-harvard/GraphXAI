import sys
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from graphxai.datasets import ShapeGraph

try:
    nclusters = int(sys.argv[1])
except:
    print('usage: python3 test_clusters.py <n_clusters_per_class>')
    exit()

SG = ShapeGraph(
    model_layers = 3,
    make_explanations=False,
    num_subgraphs = 100,
    prob_connection = 0.08,
    subgraph_size = 12,
    class_sep = 2,
    n_informative = 6,
    n_clusters_per_class = nclusters,
    verify = False
)

data = SG.get_graph()

tsne_X = TSNE().fit_transform(data.x.numpy())
Y = data.y.numpy()

plt.scatter(tsne_X[:,0], tsne_X[:,1], c = Y, cmap = 'winter')
plt.title('NC = {}'.format(nclusters))
plt.show()
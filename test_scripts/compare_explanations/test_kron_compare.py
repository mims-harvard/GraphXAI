import torch

from torch_geometric.data import Data
from torch_geometric.utils import convert
from networkx.algorithms.swap import double_edge_swap as swap
from networkx.linalg.graphmatrix import adjacency_matrix as adj_mat

from IPython.display import SVG
from sknetwork.visualization import svg_graph

from old import kronecker_graph


def rewire_edges(x, edge_index, degree):
    # Convert to networkx graph for rewiring edges
    data = Data(x=x, edge_index=edge_index)
    G = convert.to_networkx(data, to_undirected=True)
    rewired_G = swap(G, nswap=degree, max_tries=degree * 25, seed=912)
    rewired_adj_mat = adj_mat(rewired_G)
    rewired_edge_indexes = convert.from_scipy_sparse_matrix(rewired_adj_mat)[0]
    return rewired_edge_indexes

K1 = torch.tensor([[1, 1, 1, 1],
                   [1, 1, 0, 0],
                   [1, 0, 1, 0],
                   [1, 0, 0, 1]])
edge_index = kronecker_graph(K1, 2)
num_nodes = 4 ** 2
x = torch.randn(num_nodes, 1)

adj = convert.to_scipy_sparse_matrix(edge_index).tocsr()
img = svg_graph(adj)
SVG(img)

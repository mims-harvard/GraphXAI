import torch
import networkx as nx
import matplotlib.pyplot as plt

from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx


def visualize_explanation(edge_index: torch.Tensor, num_nodes: int = None,
                          node_imp: list = None, edge_imp: list = None):
    """
    Visualize node_imp with node color and edge_imp with edge alpha.
    """
    if num_nodes is None:
        data = Data(edge_index=edge_index)
    else:
        data = Data(edge_index=edge_index, num_nodes=num_nodes)

    G = to_networkx(data)

    if node_imp is None:
        node_imp = [0.9 for _ in range(data.num_nodes)]
    if edge_imp is None:
        edge_imp = [0.1 for _ in range(edge_index.shape[1])]

    pos = nx.kamada_kawai_layout(G)
    nx.draw(G, pos,
            cmap=plt.get_cmap('viridis'),
            node_color=node_imp,
            width=edge_imp)
    plt.show()

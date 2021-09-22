import torch
import networkx as nx
import matplotlib.pyplot as plt

from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx


def visualize_edge_explanation(edge_index: torch.Tensor, node_idx: int = None,
                               num_nodes: int = None, edge_imp: list = None):
    """
    Visualize edge_imp with edge alpha.

    node_idx is the node explained.
    """
    if num_nodes is None:
        data = Data(edge_index=edge_index)
    else:
        data = Data(edge_index=edge_index, num_nodes=num_nodes)

    G = to_networkx(data, to_undirected=True)

    if edge_imp is None:
        edge_imp = [0.1 for _ in range(edge_index.shape[1])]

    pos = nx.kamada_kawai_layout(G)

    if node_idx is None:
        nx.draw_networkx_nodes(G, pos)
    else:
        nx.draw_networkx_nodes(G, pos, nodelist=[node_idx],
                               node_color="tab:red")
        nx.draw_networkx_nodes(G, pos, nodelist=set(G.nodes) - {node_idx},
                               node_color="tab:blue")

    nx.draw_networkx_edges(G, pos, width=edge_imp)

    plt.axis('off')
    plt.grid(False)
    plt.show()

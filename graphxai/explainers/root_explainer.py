import torch
import torch.nn as nn
from torch_geometric.utils import k_hop_subgraph


class RootExplainer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    # @property
    # """
    # add different util functions that may be used by differnet explanation methods
    # """

    def get_explanation_node(self, *args, **kwargs):
        """
        Explain a node prediction
        """
        raise NotImplementedError()

    def get_explanation_graph(self):
        """
        Explain a graph prediction
        """
        raise NotImplementedError()

    def get_explanation_link(self):
        """
        Explain an edge link prediction
        """
        raise NotImplementedError()

    def get_subgraph(self, node_idx, num_hops, x, edge_index):
        subset, sub_edge_index, mapping, hard_edge_mask = k_hop_subgraph(
            node_idx, num_hops, edge_index, relabel_nodes=True, num_nodes=x.size(0))
        sub_x = x[subset]
        sub_num_edges = sub_edge_index.size(1)
        sub_num_nodes, num_features = sub_x.size()
        return subset, sub_edge_index

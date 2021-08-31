import torch

from torch_geometric.utils.random import stochastic_blockmodel_graph


def sbm_with_singletons(block_sizes: list, num_singletons: int,
                        p_in: list, p_out: float):
    b = len(block_sizes) + num_singletons
    block_sizes = torch.tensor(block_sizes + num_singletons * [1])
    p_in = torch.tensor(p_in + num_singletons * [0])  # No self-loop in singletons

    edge_probs = torch.ones(b, b) * p_out
    edge_probs[torch.eye(b).bool()] = p_in

    edge_index = stochastic_blockmodel_graph(block_sizes, edge_probs, directed=False)
    return edge_index

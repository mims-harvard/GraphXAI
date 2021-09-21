import torch


def kronecker_graph(K1, num_powers, edge_prob=0.5):
    """
    K1 is the adjacency matrix of the initiator graph. K1 has self-loops.
    """
    assert len(K1.shape) == 2 and K1.shape[0] == K1.shape[1]
    K = K1
    for _ in range(1, num_powers):
        K = torch.kron(K, K1)
    # Remove self-loops
    K.fill_diagonal_(0)
    K = torch.bernoulli(edge_prob * K)
    edge_index = K.nonzero().T
    return edge_index

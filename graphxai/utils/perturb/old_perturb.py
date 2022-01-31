import torch

from typing import List
from torch_geometric.data import Data
from torch_geometric.utils import convert
from torch_geometric.utils.subgraph import k_hop_subgraph
from networkx.linalg.graphmatrix import adjacency_matrix as adj_mat

from .nx_modified import swap


def rewire_edges(edge_index: torch.Tensor, num_nodes: int,
                 node_idx: int = None, num_hops: int = 3,
                 rewire_prob: float = 0.01, seed: int = 0):
    """
    Rewire edges in the graph.

    If subset is None, rewire the whole graph.
    Otherwise, rewire the edges within the induced subgraph of subset of nodes.
    """
    # Get the k-hop subgraph of node_idx if specified, and compute nswap
    if node_idx is None:
        subset = None
        m = edge_index.shape[1]
        nswap = round(m*rewire_prob)
    else:
        subset, sub_edge_index, _, _ = k_hop_subgraph(node_idx, num_hops, edge_index)
        m = sub_edge_index.shape[1]
        nswap = round(m*rewire_prob)

    # Convert to networkx graph for rewiring edges
    data = Data(edge_index=edge_index, num_nodes=num_nodes)
    G = convert.to_networkx(data, to_undirected=True)
    rewired_G = swap(G, subset, nswap=nswap, max_tries=500*nswap, seed=seed)
    rewired_adj_mat = adj_mat(rewired_G)
    rewired_edge_index = convert.from_scipy_sparse_matrix(rewired_adj_mat)[0]
    return rewired_edge_index


def PGM_perturb_node_features(x: torch.Tensor, perturb_prob: float = 0.5,
                          bin_dims: List[int] = [], perturb_mode: str = 'scale'):
    """
    Pick nodes with probability perturb_prob and perturb their features.

    The continuous dims are assumed to be non-negative.
    The discrete dims are required to be binary.

    Args:
        x (torch.Tensor, [n x d]): node features
        perturb_prob (float): the probability that a node in node_idx is perturbed
        bin_dims (list of int): the list of binary dims
        perturb_mode (str):
            'scale': randomly scale (0-2) the continuous dim
            'gaussian': add a Gaussian noise to the continuous dim
            'uniform': add a uniform noise to the continuous dim
            'mean': set the continuous dims to their mean value

    Returns:
        x_pert (torch.Tensor, [n x d]): perturbed feature matrix
        node_mask (torch.Tensor, [n]): Boolean mask of perturbed nodes
    """
    n ,d = x.shape
    cont_dims = [i for i in range(d) if i not in bin_dims]
    c = len(cont_dims)
    b = len(bin_dims)

    # Select nodes to perturb
    pert_mask = torch.bernoulli(perturb_prob * torch.ones(n)).bool()
    nodes_pert = pert_mask.nonzero().flatten()
    n_pert = nodes_pert.shape[0]
    x_new = x.clone()
    x_pert = x_new[nodes_pert]

    max_val, _ = torch.max(x[:, cont_dims], dim=0, keepdim=True)

    if perturb_mode == 'scale':
        # Scale the continuous dims randomly
        x_pert[:, cont_dims] *= 2 * torch.rand(c)
    elif perturb_mode == 'gaussian':
        # Add a Gaussian noise
        sigma = torch.std(x[:, cont_dims], dim=0, keepdim=True)
        x_pert[:, cont_dims] += sigma * torch.randn(size=(n_pert, c))
    elif perturb_mode == 'uniform':
        # Add a uniform noise
        epsilon = 0.05 * max_val
        x_pert[:, cont_dims] += 2*epsilon * (torch.rand(size=(n_pert, c)) - 0.5)
    elif perturb_mode == 'mean':
        # Set to mean values
        mu = torch.mean(x[:, cont_dims], dim=0, keepdim=True)
        x_pert[:, cont_dims] = mu
    else:
        raise ValueError("perturb_mode must be one of ['scale', 'gaussian', 'uniform', 'mean']")

    # Ensure feature value is between 0 and max_val
    x_pert[:, cont_dims] = torch.clamp(x_pert[:, cont_dims],
                                       min=torch.zeros(1, c), max=max_val)

    # Randomly flip the binary dims
    x_pert[:, bin_dims] = torch.randint(2, size=(n_pert, b)).float()

    # Copy x_pert to perturbed nodes in x_new
    x_new[nodes_pert] = x_pert

    return x, pert_mask

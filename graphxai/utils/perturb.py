import torch

from typing import List


def perturb_node_features(x: torch.Tensor, node_idx: torch.Tensor = None,
                          perturb_prob: float = 0.5, cont_dims: List[int] = []):
    """
    Select each node in node_idx with perturb_prob and perturb the nodes selected.

    The discrete dims are required to be binary and flipped with perturb_prob.
    The continuous dims are added a scaled Gaussian white noise.

    Args:
        x (torch.Tensor, [n x d]): node features
        node_idx (torch.Tensor, optional, [k]): the indices of the k nodes interested
            If not provided, include all nodes by default.
        perturb_prob (float): the probability that a node in node_idx is perturbed
        cont_dims (list of int): the list of continuous dims

    Returns:
        x_pert (torch.Tensor, [n x d]): perturbed feature matrix
        node_mask (torch.Tensor, [n]): Boolean mask of perturbed nodes
    """
    n ,d = x.shape
    x_pert = x.clone()

    # Only consider perturbing nodes in node_idx
    if node_idx is None:
        node_mask = torch.ones(n)
    else:
        node_mask = torch.zeros(n)
        node_mask[node_idx] = 1
    # Compute the perturbation mask of nodes
    node_mask = torch.bernoulli(perturb_prob * node_mask)

    # Compute the (scaled) feature dim mask for continuous dims
    cont_scale = torch.zeros(d)
    sigma = torch.std(x[:, cont_dims], dim=0)
    cont_scale[cont_dims] = sigma
    cont_pert = torch.outer(node_mask, cont_scale)
    cont_mask = cont_pert.bool()

    # Perturb the continuous dims
    cont_pert_slice = cont_pert[cont_mask]
    cont_pert[cont_mask] *= torch.randn_like(cont_pert_slice)
    x_pert += cont_pert

    # Flip the binary dims
    bin_mask = torch.outer(node_mask, ~cont_scale.bool()).bool()
    x_pert[bin_mask] = 1 - x_pert[bin_mask]

    return x_pert, node_mask.bool()

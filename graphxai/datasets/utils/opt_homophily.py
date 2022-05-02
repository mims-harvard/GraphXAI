import random
import torch
import torch.nn.functional as F

def if_edge_exists(edge_index: torch.Tensor, node1: int, node2: int):
    '''
    Quick lookup for if an edge exists b/w `node1` and `node2`
    '''
    
    p1 = torch.any((edge_index[0,:] == node1) & (edge_index[1,:] == node2))
    p2 = torch.any((edge_index[1,:] == node1) & (edge_index[0,:] == node2))

    return (p1 or p2).item()

def optimize_homophily(
        x: torch.Tensor, 
        edge_index: torch.Tensor,
        label: torch.Tensor,
        feature_mask: torch.Tensor, 
        homophily_coef: float = 1.0, 
        epochs: int = 50, 
        connected_batch_size: int = 10,
        disconnected_batch_size: int = 10,
    ):
    '''
    Optimizes the graph features to have a set level of homophily or heterophily

    Args:
        x (torch.Tensor): Initial node features. `|V| x d` tensor, where `|V|` is number of nodes,
            `d` is dimensionality of each node feature.
        edge_index (torch.Tensor): Edge index. Standard `2 x |E|` shape.
        label (torch.Tensor): All node labels. Shape `|V|,` tensor.
        feature_mask (torch.Tensor): Boolean tensor over the dimensions of each feature. Tensor
            should be size `d,`.
        homophily_coef (float, optional): Homophily coefficient on which to optimize the level 
            of homophily or heterophily in the graph. Positive values indicate homophily while
            negative values indicate heterophily. (:default: :obj:`1.0`)
        epochs (int, optional): Number of epochs on which to optimize features. (:default: :obj:`50`)
        connected_batch_size (int, optional): Batch size at each epoch for connected nodes on which 
            to observe for the loss function. (:default: :obj:`10`) 
        disconnected_batch_size (int, optional): Batch size at each epoch for disconnected nodes on which 
            to observe for the loss function. (:default: :obj:`10`) 

    :rtype: `torch.Tensor`
    Returns:
        x (torch.Tensor): Optimized node features.
    '''

    to_opt = x.detach().clone()[:,feature_mask]

    optimizer = torch.optim.Adam([to_opt], lr=0.3)
    to_opt.requires_grad = True

    # Get indices for connected nodes having same label
    c_inds = torch.randperm(edge_index.shape[1])[:connected_batch_size]
    c_inds = c_inds[label[edge_index.t()[c_inds][:, 0]] == label[edge_index.t()[c_inds][:, 1]]]

    # Get indices for connected nodes having different label
    nc_inds = torch.randperm(edge_index.shape[1])[:connected_batch_size]
    nc_inds = nc_inds[label[edge_index.t()[nc_inds][:, 0]] != label[edge_index.t()[nc_inds][:, 1]]]

    # Get set of nodes that are either connected or not connected, with different labels:
    # [[a1, a2, a3, ...], [b1, b2, b3, ...]]
    nc_list1 = torch.full((disconnected_batch_size,), -1) # Set to dummy values in beginning
    nc_list2 = torch.full((disconnected_batch_size,), -1)

    nodes = torch.arange(x.shape[0])

    for i in range(disconnected_batch_size):
        c1, c2 = random.choice(nodes).item(), random.choice(nodes).item()

        # Get disconnected and with same label
        while if_edge_exists(edge_index, c1, c2) or \
                torch.any((nc_list1 == c1) & (nc_list2 == c2)) or \
                (label[c1] != label[c2]).item():
            c1, c2 = random.choice(nodes).item(), random.choice(nodes).item()

        # Fill lists if we found valid choice:
        nc_list1[i] = c1
        nc_list2[i] = c2

        # May be problems with inifinite loops with large batch sizes
        #   - Should control upstream to avoid

    for i in range(epochs):
        # Compute similarities for all edges in the c_inds:
        c_cos_sim = F.cosine_similarity(to_opt[edge_index.t()[c_inds][:, 0]], to_opt[edge_index.t()[c_inds][:, 1]])
        nc_cos_sim = F.cosine_similarity(to_opt[nc_list1], to_opt[nc_list2])
        diff_label_sim = F.cosine_similarity(to_opt[edge_index.t()[nc_inds][:, 0]], to_opt[edge_index.t()[nc_inds][:, 1]])
        optimizer.zero_grad()
        loss = -homophily_coef * c_cos_sim.mean() + (homophily_coef)*(nc_cos_sim.mean() + diff_label_sim.mean())
        #loss = -homophily_coef * c_cos_sim.mean() + ((1 - homophily_coef) / 2) * (nc_cos_sim.mean() + diff_label_sim.mean())
        loss.backward()
        optimizer.step()

    # Assign to appropriate copies:
    xcopy = x.detach().clone()
    xcopy[:,feature_mask] = to_opt.detach().clone()

    return xcopy
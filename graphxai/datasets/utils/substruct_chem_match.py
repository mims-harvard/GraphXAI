import torch

from torch_geometric.utils import k_hop_subgraph, subgraph

def match_NO2(data):
    '''
    Identifies edges and nodes in a graph that correspond to NO2 groups

    Args:
        data (torch_geometric.data.Data): One graph on which to match for
            NO2 groups

    Returns:
        List of (subgraph nodes (Tensor), edge mask (Tensor))
    '''

    isN = []
    for i in range(data.x.shape[0]):
        # We know that index 1 in x vec is N (one-hot)
        if data.x[i,1].item() == 1:
            isN.append(i)

    # Get k-hop subgraph around all N's
    # Check if that subgraph contains two O's with only two 
    #   edges in the edge_index (since its undirected)

    ground_truths = []

    for N in isN:
        subset, _, _, _ = k_hop_subgraph()

        # 1. make sure there's two O's:
        Os = []
        for sub_node in subset.tolist():
            # We know that index 2 in x is O (one-hot)
            if data.x[sub_node,2].item() == 1:
                Os.append(sub_node)

        if len(Os) != 2:
            # Needs to have two O's
            break

        # Examine the Os:
        Os_pass = True
        for O in Os:
            # Count the number of occurences in each row of 
            #   the edge index:
            num_0 = torch.sum(data.edge_index[0,:] == O).item()
            num_1 = torch.sum(data.edge_index[1,:] == O).item()

            if not (num_0 == num_1 == 1):
                Os_pass = False # Set flag

        if Os_pass: # Know that we have a hit
            subgraph_nodes = torch.tensor([N] + Os, dtype = int)

            _, _, edge_mask = subgraph(
                subgraph_nodes, 
                edge_index = data.edge_index,
                return_edge_mask = True,
                )

            ground_truths.append((subgraph_nodes, edge_mask))

    return ground_truths
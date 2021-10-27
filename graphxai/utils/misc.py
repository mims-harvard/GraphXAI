import torch

def make_node_ref(nodes: torch.tensor):
    '''
    Makes a node reference to unite node indicies across explanations
    Args:
        nodes (torch.tensor): Tensor of nodes to reference.
    '''
    node_reference = {nodes[i].item():i for i in range(nodes.shape[0])}
    return node_reference

# def make_edge_ref():
#     pass
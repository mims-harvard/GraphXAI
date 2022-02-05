import os
import torch 

def exp_exists(node_idx, path = '', get_exp = False):
    '''
    If the explanation exists, return the explanation
        else, return None
    '''

    if isinstance(node_idx, torch.Tensor):
        node_idx = node_idx.item()

    fname = 'exp_node{:0<5d}.pt'.format(node_idx)
    
    full_path = os.path.join(path, fname)

    if get_exp:
        if os.path.exists(full_path):
            return None
        else:
            # Get the explanation:
            t = torch.load(open(full_path, 'rb'))
            return t
    else:
        return os.path.exists(full_path)

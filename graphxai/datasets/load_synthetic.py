import os
import torch

root_path = '~/GraphXAI/data/ShapeGraph/'

# def load_ShapeGGennumber: int = 1, root: str = Owen_root):
#     '''
#     Args:
#         number (int): Number of ShapeGraph to load.
#         root (str): Root to directory containing all saved ShapeGraphs.
#     '''
#     return torch.load(open(os.path.join(root, 'ShapeGraph_{}.pickle'.format(number)), 'rb'))

def load_ShapeGGenname, root = root_path):
    return torch.load(open(os.path.join(root, name), 'rb'))

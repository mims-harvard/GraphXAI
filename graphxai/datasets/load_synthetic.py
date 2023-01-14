import os
import torch

root_path = '~/GraphXAI/data/ShapeGGen/'

# def load_ShapeGGen(number: int = 1, root: str = Owen_root):
#     '''
#     Args:
#         number (int): Number of ShapeGGen to load.
#         root (str): Root to directory containing all saved ShapeGGens.
#     '''
#     return torch.load(open(os.path.join(root, 'ShapeGGen_{}.pickle'.format(number)), 'rb'))

def load_ShapeGGen(name, root = root_path):
    return torch.load(open(os.path.join(root, name), 'rb'))

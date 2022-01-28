import os
import torch

Owen_root = '/Users/owenqueen/Desktop/HMS_research/graphxai_project/GraphXAI/data/ShapeGraph/unzipped'

# def load_ShapeGraph(number: int = 1, root: str = Owen_root):
#     '''
#     Args:
#         number (int): Number of ShapeGraph to load.
#         root (str): Root to directory containing all saved ShapeGraphs.
#     '''
#     return torch.load(open(os.path.join(root, 'ShapeGraph_{}.pickle'.format(number)), 'rb'))

def load_ShapeGraph(name, root = Owen_root):
    return torch.load(open(os.path.join(root, name, 'rb')))

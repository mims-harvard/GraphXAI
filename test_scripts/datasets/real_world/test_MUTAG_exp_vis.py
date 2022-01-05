import random

import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from torch_geometric.utils import to_networkx
from graphxai.datasets.real_world.MUTAG import MUTAG

mutag = MUTAG(root = '.')

for i in random.sample(list(range(len(mutag))), k = 50):
    g, exp = mutag[i]

    exp.graph_draw(show = True)

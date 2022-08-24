import torch
import numpy as np
from graphxai.datasets import AlkaneCarbonyl

dataset = AlkaneCarbonyl(split_sizes = (0.7, 0.2, 0.1), seed = 1234)

print('Class imbalance:')
print('Label==0:', np.sum([dataset.graphs[i].y.item() == 0 for i in range(len(dataset))]))
print('Label==1:', np.sum([dataset.graphs[i].y.item() == 1 for i in range(len(dataset))]))

import random
y_mask = torch.as_tensor([dataset.graphs[i].y.item() == 1 for i in range(len(dataset))]).nonzero(as_tuple=True)[0]

exp_list = dataset.explanations[y_mask[0].item()]

for exp in exp_list:
    exp.visualize_graph(show=True)
import torch
import numpy as np
from graphxai.datasets import FluorideCarbonyl

dataset = FluorideCarbonyl(split_sizes = (0.7, 0.2, 0.1), seed = 1234)

print('Class imbalance:')
print('Label==0:', np.sum([dataset.graphs[i].y.item() == 0 for i in range(len(dataset))]))
print('Label==1:', np.sum([dataset.graphs[i].y.item() == 1 for i in range(len(dataset))]))
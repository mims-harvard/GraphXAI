import torch

from graphxai.datasets import Benzene

dataset = Benzene(split_sizes = (0.7, 0.2, 0.1), seed = 1234)

print(dataset.test_index)
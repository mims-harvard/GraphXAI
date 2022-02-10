import torch

from graphxai.datasets import Mutagenicity

dataset = Mutagenicity(root = '../../explainers/mutag/data', split_sizes = (0.8, 0.1, 0.1), seed = 1234)

print(dataset.test_index)

print(dataset.test_index.shape)
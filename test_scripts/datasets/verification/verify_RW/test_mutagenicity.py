import torch
from tqdm import trange
from graphxai.datasets import Mutagenicity

dataset = Mutagenicity(root = '../../../explainers/mutagenicity/data')

# Iterate through dataset, check for null explanations

conf_matrix = torch.zeros((2, 2))

for i in trange(len(dataset.graphs)):
    matches = int(dataset.explanations[i][0].has_match)
    yval = int(dataset.graphs[i].y.item())

    conf_matrix[matches, yval] += 1

print(conf_matrix)
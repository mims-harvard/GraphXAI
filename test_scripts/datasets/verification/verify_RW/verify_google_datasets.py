import torch
from tqdm import trange
from graphxai.datasets import Benzene, FluorideCarbonyl

dataset = FluorideCarbonyl(split_sizes = (0.8, 0.2, 0), seed = 1234)
conf_matrix = torch.zeros((2, 2))

for i in trange(len(dataset)):
    data, exp = dataset[i]
    yval = int(data.y)

    conf_matrix[int(exp[0].has_match), yval] += 1

print('Confusion matrix')
print(conf_matrix)
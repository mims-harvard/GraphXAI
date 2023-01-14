import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_edge_index(fname):
    L = []
    with open(fname, 'r') as f:
        for l in f:
            line = [float(g) for g in l.split()]
            L.append(line)
        #L.append()

    edge_index = torch.tensor(L).transpose(0,1)

    return edge_index

def degdist(edge_index):

    nodes = torch.unique(edge_index)
    dist = []
    for n in tqdm(nodes):
        num = (edge_index[0,:] == n).sum().item()
        dist.append(num)

    return dist


if __name__ == '__main__':
    e = get_edge_index('/Users/owenqueen/Desktop/HMS_research/graphxai_project/GraphXAI/data/RW/german/german_edges.txt')
    print(e.shape)

    degdist(e)

    
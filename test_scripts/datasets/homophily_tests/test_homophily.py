import os, random
import torch
import numpy as np
import torch.nn.functional as F
#from tqdm import trange
import matplotlib.pyplot as plt

from torch_geometric.utils import sort_edge_index, to_undirected
from graphxai.datasets import ShapeGraph
from graphxai.datasets.load_synthetic import load_ShapeGraph

def if_edge_exists(edge_index, node1, node2):
    
    p1 = torch.sum((edge_index[0,:] == node1) & (edge_index[1,:] == node2)).bool()
    p2 = torch.sum((edge_index[1,:] == node1) & (edge_index[0,:] == node2)).bool()

    return (p1 or p2).item()

def homophily_test(SG: ShapeGraph, k_sample: int = 1000, label = 0):
    
    data = SG.get_graph()
    k = min(data.edge_index.shape[1], k_sample)

    label_mask = (data.y == label)
    #x = data.x[label_mask]

    nonzero_nodes = label_mask.nonzero(as_tuple = True)[0]

    eidx_mask0 = torch.as_tensor([label_mask[n] for n in data.edge_index[0,:]]).bool()
    eidx_mask1 = torch.as_tensor([label_mask[n] for n in data.edge_index[1,:]]).bool()

    #eidx_mask1 = torch.as_tensor([torch.sum(data.edge_index[1,:] == n).bool() for n in nonzero_nodes])


    #eidx_mask = torch.as_tensor([(torch.sum(data.edge_index[0,:] == n).bool() and torch.sum(data.edge_index[1,:] == n)) for n in label_mask.nonzero(as_tuple = True)[0]])

    eidx_mask = eidx_mask0 & eidx_mask1

    edge_index = data.edge_index[:,eidx_mask]

    connected_cosine = []
    not_connected_cosine = []

    # Get nodes that are connected
    c_inds = torch.randperm(edge_index.shape[1])[:k]

    # Compute similarities for all edges in the c_inds:
    for n1, n2 in edge_index.t()[c_inds]:
        #print('Label 1 and 2: {}, {}'.format(data.y[n1], data.y[n2]))
        cos = F.cosine_similarity(data.x[n1,:], data.x[n2,:], dim = 0)
        connected_cosine.append(cos.item())

    # Get random set of combinations 
    #nc_inds = torch.randperm(data.x.shape[0])[:k]
    #combs = torch.combinations(torch.arange(data.x.shape[0]), r = 2, with_replacement = True)

    combs = torch.combinations(label_mask.nonzero(as_tuple=True)[0], r = 2, with_replacement=True)
    print('finished combs')
    #possible_inds = set(range(combs.shape[0]))

    chosen_inds = []

    targ = torch.arange(combs.shape[0])

    for i in range(k):
        pi = random.choice(targ)
        random_pair = combs[pi]
        
        # Go until we get one:
        while if_edge_exists(data.edge_index, random_pair[0], random_pair[1]): #and (pi in chosen_inds):
            pi = random.choice(targ)
            random_pair = combs[pi]
        chosen_inds.append(pi)

    print('Finish first loop')
    for pi in chosen_inds:

        random_pair = combs[pi]

        n1, n2 = random_pair[0], random_pair[1]
        #print('Label 1 and 2: {}, {}'.format(data.y[n1], data.y[n2]))
        cos = F.cosine_similarity(data.x[n1,:], data.x[n2,:], dim = 0)
        not_connected_cosine.append(cos.item())

    print('Finish last loop')

    return connected_cosine, not_connected_cosine


if __name__ == '__main__':
    SG = ShapeGraph(
        model_layers = 3,
        num_subgraphs = 100,
        prob_connection = 0.08,
        subgraph_size = 10,
        class_sep = 5,
        n_informative = 6,
        n_clusters_per_class = 1,
        verify = False,
        make_explanations=False
    )

    con, ncon = homophily_test(SG, k_sample = 1000)

    print('Mean cosine similarity of connected nodes:', np.mean(con))
    print('Mean cosine similarity of disconnected nodes:', np.mean(ncon))

    plt.boxplot([con, ncon])
    plt.xticks(ticks = [1, 2], labels = ['Connected', 'Not Connected'])
    plt.ylabel('Cosine similarity')
    plt.title('Homophily test, separation = 5')
    plt.show()

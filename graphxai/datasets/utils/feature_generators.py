import random
import torch
import networkx as nx

from graphxai.datasets.feature import make_structured_feature

def net_stats_generator(G):
    '''
    Args:
        G (nx.Graph): 
    '''
    deg_cent = nx.degree_centrality(G)
    def get_feature(node_idx):
        return torch.tensor([G.degree[node_idx], 
            nx.clustering(G, node_idx), 
            deg_cent[node_idx]]).float()

    return get_feature

def random_continuous_generator(len_vector = 3):
    '''
    Generate random continuous vectors of length given
    Args:
        len_vector (int): Length of vectors to generate

    :rtype: Callable[[int], torch.Tensor]
        - Return is a function that takes integer and outputs vector
    '''
    def get_feature(node_idx):
        # Random random Gaussian feature vector:
        return torch.normal(mean=0, std=1.0, size = (len_vector,))
    return get_feature

def random_onehot_generator(len_vector):
    '''
    Generate random onehot vectors of length given
    Args:
        len_vector (int): Length of vectors to generate

    :rtype: Callable[[int], torch.Tensor]
        - Return is a function that takes integer and outputs vector
    '''
    def get_feature(node_idx):
        # Random one-hot feature vector:
        feature = torch.zeros(3)
        feature[random.choice(range(3))] = 1
        return feature
    return get_feature

def gaussian_lv_generator(
        G: nx.Graph, 
        yvals: torch.Tensor,  
        n_features: int = 10,       
        flip_y: float = 0.01,
        class_sep: float = 1.0,
        n_informative: int = 4,
        n_clusters_per_class: int = 2,
        seed = None):
    '''
    Args:
        G (nx.Graph): 
        yvals (torch.Tensor): 
        seed (seed): (:default: :obj:`None`)
    '''

    x, feature_imp_true = make_structured_feature(
            yvals, 
            n_features = n_features,
            n_informative = n_informative, 
            flip_y = flip_y,
            class_sep=class_sep,
            n_clusters_per_class=n_clusters_per_class,
            seed = seed)

    Gitems = list(G.nodes.items())
    node_map = {Gitems[i][0]:i for i in range(G.number_of_nodes())}

    def get_feature(node_idx):
        return x[node_map[node_idx],:]

    return get_feature, feature_imp_true

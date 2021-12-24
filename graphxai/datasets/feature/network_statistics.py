import torch
import networkx as nx

from torch_geometric.data import Data
from torch_geometric.utils import degree
from torch_geometric.utils.convert import to_networkx
from torch_geometric.utils.num_nodes import maybe_num_nodes


def make_network_stats_feature(edge_index: torch.Tensor, include: list = ['degree'],
                               num_useless_features=0, shuffle=True, normalize=True,
                               num_nodes=None):
    N = maybe_num_nodes(edge_index, num_nodes)
    feature_names = []
    features = []
    for feature_name in include:
        if feature_name == 'degree' or feature_name == 'out_degree':
            out_degree = degree(edge_index[0], N)
            features.append(out_degree.view(-1, 1))
        elif feature_name == 'in_degree':
            in_degree = degree(edge_index[1], N)
            features.append(in_degree.view(-1, 1))
        elif feature_name == 'clustering_coefficient':
            G = to_networkx(Data(edge_index=edge_index, num_nodes=N))
            cc = torch.tensor(list(nx.clustering(G).values()))
            features.append(cc.view(-1, 1))
        else:
            continue
        feature_names.append(feature_name)

    X = torch.cat(features)

    # Assume all non-noise features are useful
    feature_mask = torch.zeros(len(features)+num_useless_features, dtype=bool)
    feature_mask[:len(features)] = True

    # Fill useless features with noise
    if num_useless_features > 0:
        noise = torch.randn(N, num_useless_features)
        feature_names += ['noise'] * num_useless_features
        X = torch.cat([X, noise], dim=-1)

    if shuffle:
        # Randomly permute features
        order = torch.randperm(X.shape[1])
        X = X[:, order]
        feature_mask[:] = feature_mask[order]
        feature_names = [feature_names[i] for i in order]

    # Normalize feature
    if normalize:
        X = X / torch.norm(X, dim=0)

    return X, feature_mask, feature_names

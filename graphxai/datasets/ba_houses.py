import torch
import random
import numpy as np
import networkx as nx
from typing import Optional
from functools import partial

from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

from sklearn.model_selection import train_test_split

from .dataset import NodeDataset
from graphxai.utils import Explanation
from .ba_houses_generators import generate_BAHouses_graph_global, generate_BAHouses_graph_local

class BAHouses(NodeDataset):
    '''
    Generate a Barabasi-Albert graph with planted house motifs.

    Args:
        num_hops (int): Number of hops in each node's computational graph.
            Corresponds to number of convolutional layers in GNN.
        n (int): For global planting method, corresponds to the total number of 
            nodes in graph. If using local planting method, corresponds to the 
            starting number of nodes in graph.
        m (int): Number of edges per node in graph.
        num_houses (int, optional): Number of houses to add in entire graph.
            Only active for global planting method, i.e. `plant_method == 'global'`.
            (:default: :obj:`None`)
        k (int, optional): Lower-bound on number of houses per neighborhood.
            Only active for local planting method, i.e. `plant_method == 'local'`.
            (:default: :obj:`None`)
        seed (int, optional): Seed for random generation of graph. (:default: `None`)
        plant_method (str, optional): Options are `'local'` and `'global'`, corresponding
            to global and local planting methods, respectfully. (:default: `'global'`)
        kwargs:
            - in_hood_numbering (bool, optional): y labels become number of houses in 
                L-hop neighborhood.
            - threshold (int, optional): Threshold for 0 or 1 assignment of house labels.
                For all nodes, if `node.num_houses <= threshold`, y value assigned to 0.
                Else, the y value is assigned to 1. 
    '''

    def __init__(self, 
        num_hops: int, 
        n: int, 
        m: int, 
        num_houses: int = None, 
        k: int = None,
        seed: Optional[int] = None,
        plant_method: Optional[str] = 'global',
        **kwargs):

        super().__init__(name='BAHouses', num_hops=num_hops)
        self.n = n
        self.m = m
        self.seed = seed

        if plant_method == 'global':
            self.generate_data = partial(
                generate_BAHouses_graph_global,
                n = n, 
                m = m, 
                num_houses = num_houses, 
                num_hops = num_hops, 
                seed = seed,
                get_data = True,
                **kwargs
            )

        elif plant_method == 'local':
            self.generate_data = partial(
                generate_BAHouses_graph_local,
                n = n,
                m = m, 
                k = k,
                num_hops = num_hops,
                seed = seed,
                get_data = True,
                **kwargs
            )

        # Generate data:
        self.graph, self.explanations = self.generate_data()

        # Formulate static split (independent of seed):
        # Keep seed constant for reproducible splits
        train_mask, rem_mask = train_test_split(list(range(self.graph.num_nodes)), 
                                test_size = 0.3, 
                                random_state = 1234)

        valid_mask, test_mask = train_test_split(rem_mask, 
                                test_size = 1.0 / 3,
                                random_state = 5678)

        self.fixed_train_mask = torch.tensor([i in train_mask for i in range(self.graph.num_nodes)], dtype = torch.bool)
        self.fixed_valid_mask = torch.tensor([i in valid_mask for i in range(self.graph.num_nodes)], dtype = torch.bool)
        self.fixed_test_mask  = torch.tensor([i in test_mask  for i in range(self.graph.num_nodes)], dtype = torch.bool)
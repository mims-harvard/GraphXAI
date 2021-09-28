import torch
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from types import MethodType
from typing import Optional, Callable
from torch_geometric.utils import k_hop_subgraph

from ..synthetic_dataset import ShapeGraph
from graphxai.utils.nx_conversion import khop_subgraph_nx
from graphxai import Explanation
from ..utils.shapes import house

class BAHouses(ShapeGraph):
    '''
    BA Houses

    Args:
        num_hops (int): Number of hops for each node's enclosing 
            subgraph. Should correspond to number of graph convolutional
            layers in GNN. 
        num_houses (int): Number of shapes for given planting strategy.
            If planting strategy is global, total number of shapes in
            the graph. If planting strategy is local, number of shapes
            per num_hops - hop neighborhood of each node.
        insert_method (str, optional): Type of insertion strategy for 
            each motif. Options are `'plant'` or `'staple'`.
            (:default: :obj:`'plant'`)
        plant_method (str, optional): How to decide where shapes are 
            planted. 'global' method chooses random nodes from entire 
            graph. 'local' method enforces a lower bound on number of 
            shapes in the (num_hops)-hop neighborhood of each node. 
            'neighborhood upper bound' enforces an upper-bound on the 
            number of shapes per num_hops-hop neighborhood.
            (:default: :obj:`'global'`)
        kwargs: Additional arguments
            shape_upper_bound (int, Optional): Number of maximum shapes
                to add per num_hops-hop neighborhood in the 'neighborhood
                upper bound' planting policy.
    '''

    def __init__(self, 
        num_hops: int, 
        n: int, 
        m: int, 
        num_houses: int, 
        seed: Optional[int] = None,
        insert_method: Optional[str] = 'plant',
        plant_method: Optional[str] = 'global',
        labeling_rule: Optional[Callable[[object], Callable[[int], torch.Tensor]]] = None,
        **kwargs):

        self.n = n
        self.m = m
        self.seed = seed

        self.labeling_rule = MethodType(labeling_rule, self) \
            if labeling_rule is not None else self.labeling_rule

        if plant_method == 'neighborhood upper bound' and num_houses is None:
            num_houses = n # Set to maximum if num_houses left at None

        super().__init__(
            name='BAHouses', 
            num_hops=num_hops,
            num_shapes = num_houses,
            insert_method = insert_method,
            plant_method = plant_method,
            insertion_shape = house,
            **kwargs
        )

    def init_graph(self):
        '''
        Returns a Barabasi-Albert graph with desired parameters
        '''
        self.G = nx.barabasi_albert_graph(self.n, self.m, seed = self.seed)

    def feature_generator(self):
        '''
        Returns function to generate features for one node_idx
        '''
        deg_cent = nx.degree_centrality(self.G)
        def get_feature(node_idx):
            return torch.tensor([self.G.degree[node_idx], 
                nx.clustering(self.G, node_idx), 
                deg_cent[node_idx]]).float()

        return get_feature

    def labeling_rule(self):
        '''
        Labeling rule for each node
        '''

        avg_ccs = np.mean([self.G.nodes[i]['x'][1] for i in self.G.nodes])
        
        def get_label(node_idx):
            # Count number of houses in k-hop neighborhood
            # subset, _, _, _ = k_hop_subgraph(node_idx, self.num_hops, self.graph.edge_index)
            # shapes = self.graph.shape[subset]
            # num_houses = (torch.unique(shapes) > 0).nonzero(as_tuple=True)[0]
            khop_edges = nx.bfs_edges(self.G, node_idx, depth_limit = self.num_hops)
            nodes_in_khop = set(np.unique(list(khop_edges))) - set([node_idx])
            num_unique_houses = len(np.unique([self.G.nodes[ni]['shape'] for ni in nodes_in_khop if self.G.nodes[ni]['shape'] > 0 ]))
            # Enfore logical condition:
            return torch.tensor(int(num_unique_houses == 1 and self.G.nodes[node_idx]['x'][0] > 1), dtype=torch.long)

                
        return get_label

class BAHousesRandomGaussFeatures(BAHouses):
    '''
    BA Houses dataset with random, Gaussian features for each node
    Args:
        num_hops (int): Number of hops in each node's computational graph.
            Corresponds to number of convolutional layers in GNN.
        n (int): For global planting method, corresponds to the total number of 
            nodes in graph. If using local planting method, corresponds to the 
            starting number of nodes in graph.
        m (int): Number of edges per node in graph.
        num_houses (int): If plant_method is 'global' or 'neighborhood upper bound',
            this is the number of houses to add in entire graph. If plant_method is
            'local', then this is the number of houses to add per neighborhood.
        seed (int, optional): Seed for random generation of graph. (:default: `None`)
        insert_method (str, optional): Type of insertion strategy for 
            each house. Options are `'plant'` or `'staple'`.
            (:default: :obj:`'plant'`)
        plant_method (str, optional): How to decide where houses are 
            planted. 'global' method chooses random nodes from entire 
            graph. 'local' method enforces a lower bound on number of 
            houses in the (num_hops)-hop neighborhood of each node. 
            'neighborhood upper bound' enforces an upper-bound on the 
            number of houses per num_hops-hop neighborhood.
            (:default: :obj:`'global'`)
        kwargs: Additional arguments
        
            
    '''

    def __init__(self, 
        num_hops: int, 
        n: int, 
        m: int, 
        num_houses: int, 
        seed: Optional[int] = None,
        insert_method: Optional[str] = 'plant',
        plant_method: Optional[str] = 'global',
        **kwargs):

        super().__init__(
            num_hops=num_hops,
            n = n,
            m = m,
            num_houses = num_houses,
            seed = seed,
            insert_method = insert_method,
            plant_method = plant_method,
            **kwargs
        )

    def feature_generator(self):
        '''
        Returns function to generate features for one node_idx
        '''
        def get_feature(node_idx):
            # Random random Gaussian feature vector:
            return torch.normal(mean=0, std=1.0, size = (5,))

        return get_feature

    def labeling_rule(self):
        '''
        Labeling rule for each node
        '''
        
        def get_label(node_idx):
            # Count number of houses in k-hop neighborhood
            nodes_in_khop = khop_subgraph_nx(node_idx, self.num_hops, self.G)
            num_unique_houses = len(np.unique([self.G.nodes[ni]['shape'] for ni in nodes_in_khop if self.G.nodes[ni]['shape'] > 0 ]))
            # Enfore logical condition:
            return torch.tensor(int(num_unique_houses == 1 and self.G.nodes[node_idx]['x'][2] > 0), dtype=torch.long)
                
        return get_label

    def explanation_generator(self):

        def get_explanation(node_idx):
            # Look in k-hop neighborhood:
            khop_info = k_hop_subgraph(
                node_idx = node_idx, 
                num_hops = self.num_hops, 
                edge_index=self.graph.edge_index
            )

            node_shapes = torch.tensor([ self.G.nodes[n]['shape'] for n in khop_info[0].tolist() ])
            node_imp = torch.where(node_shapes > 0, 1, 0)
            # Get nodes in subgraph
            # Find which ones are in house, if any
            
            exp = Explanation(
                feature_imp = torch.tensor([0, 0, 1, 0, 0]),
                node_imp = node_imp,
                node_idx = node_idx
            )

            exp.set_enclosing_subgraph(khop_info)
            exp.set_whole_graph(khop_info[0], khop_info[1])
            return exp

        return get_explanation

class BAHousesRandomOneHotFeatures(BAHouses):
    '''
    BA Houses dataset with random, Gaussian features for each node
    Args:
        num_hops (int): Number of hops in each node's computational graph.
            Corresponds to number of convolutional layers in GNN.
        n (int): For global planting method, corresponds to the total number of 
            nodes in graph. If using local planting method, corresponds to the 
            starting number of nodes in graph.
        m (int): Number of edges per node in graph.
        num_houses (int): If plant_method is 'global' or 'neighborhood upper bound',
            this is the number of houses to add in entire graph. If plant_method is
            'local', then this is the number of houses to add per neighborhood.
        seed (int, optional): Seed for random generation of graph. (:default: `None`)
        insert_method (str, optional): Type of insertion strategy for 
            each house. Options are `'plant'` or `'staple'`.
            (:default: :obj:`'plant'`)
        plant_method (str, optional): How to decide where houses are 
            planted. 'global' method chooses random nodes from entire 
            graph. 'local' method enforces a lower bound on number of 
            houses in the (num_hops)-hop neighborhood of each node. 
            'neighborhood upper bound' enforces an upper-bound on the 
            number of houses per num_hops-hop neighborhood.
            (:default: :obj:`'global'`)
        kwargs: Additional arguments
 
    '''

    def __init__(self, 
        num_hops: int, 
        n: int, 
        m: int, 
        num_houses: int, 
        seed: Optional[int] = None,
        insert_method: Optional[str] = 'plant',
        plant_method: Optional[str] = 'global',
        **kwargs):

        super().__init__(
            num_hops=num_hops,
            n = n,
            m = m,
            num_houses = num_houses,
            seed = seed,
            insert_method = insert_method,
            plant_method = plant_method,
            **kwargs
        )

    def feature_generator(self):
        '''
        Returns function to generate features for one node_idx
        '''
        def get_feature(node_idx):
            # Random one-hot feature vector:
            feature = torch.zeros(3)
            feature[random.choice(range(3))] = 1
            return feature

        return get_feature

    def labeling_rule(self):
        '''
        Labeling rule for each node
        '''
        
        def get_label(node_idx):
            # Count number of houses in k-hop neighborhood
            nodes_in_khop = khop_subgraph_nx(node_idx, self.num_hops, self.G)
            num_unique_houses = len(np.unique([self.G.nodes[ni]['shape'] for ni in nodes_in_khop if self.G.nodes[ni]['shape'] > 0 ]))
            # Enfore logical condition:
            return torch.tensor(int(num_unique_houses == 1 and self.G.nodes[node_idx]['x'][1] == 1), dtype=torch.long)
                
        return get_label

    def explanation_generator(self):

        def get_explanation(node_idx):
            # Look in k-hop neighborhood:
            khop_info = k_hop_subgraph(
                node_idx = node_idx, 
                num_hops = self.num_hops, 
                edge_index=self.graph.edge_index
            )

            node_shapes = torch.tensor([ self.G.nodes[n]['shape'] for n in khop_info[0].tolist() ])
            node_imp = torch.where(node_shapes > 0, 1, 0)
            # Get nodes in subgraph
            # Find which ones are in house, if any
            
            exp = Explanation(
                feature_imp = torch.tensor([0, 1, 0]),
                node_imp = node_imp,
                node_idx = node_idx
            )

            exp.set_enclosing_subgraph(khop_info)
            exp.set_whole_graph(khop_info[0], khop_info[1])
            return exp

        return get_explanation

if __name__ == '__main__':
    bah = BAHouses(
        num_hops=2,
        n=30,
        m=1,
        num_houses=1,
        plant_method='local')

    print(bah.shapes_in_graph)
    print(bah.graph.y)
    ylist = bah.graph.y.tolist()
    node_colors = [ylist[i] for i in bah.G.nodes]

    pos = nx.kamada_kawai_layout(bah.G)
    fig, ax = plt.subplots()
    nx.draw(bah.G, pos, node_color = node_colors, ax=ax)
    ax.set_title('Condition: if nhouses_in_2hop > 1 and CC < Avg(CC)')
    plt.tight_layout()
    plt.show()
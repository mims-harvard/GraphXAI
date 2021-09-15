import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from types import MethodType
from typing import Optional, Callable
from torch_geometric.utils import k_hop_subgraph

from .synthetic_dataset import ShapeGraph

def get_shape():
    '''
    Defines house in terms of a nx graph.
    '''
    G = nx.Graph() # Different from G
    nodes = list(range(5))
    G.add_nodes_from(nodes)

    # Define full cycle:
    connections = [(nodes[i], nodes[i+1]) for i in nodes[:-1]]
    connections += [(nodes[-1], nodes[0])]
    connections += [(1, 4)] # Cross-house

    G.add_edges_from(connections)
    return G

house = get_shape()

class BAHouses(ShapeGraph):
    '''
    BA Houses
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

        self.n = n
        self.m = m
        self.seed = seed

        self.labeling_rule = MethodType(kwargs['label_rule'], self) \
            if kwargs['label_rule'] is not None else self.labeling_rule

        super().__init__(
            name='BAHouses', 
            num_hops=num_hops,
            num_shapes = num_houses,
            insert_method = insert_method,
            plant_method = plant_method,
            insertion_shape = house
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
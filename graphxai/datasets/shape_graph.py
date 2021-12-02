import torch
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from types import MethodType
from typing import Optional, Callable, Union
from functools import partial
from torch_geometric.utils import k_hop_subgraph, to_undirected
from torch_geometric.data import Data

from graphxai.utils.nx_conversion import khop_subgraph_nx
from graphxai import Explanation
from graphxai.datasets.utils.shapes import *
from graphxai.datasets.dataset import NodeDataset, GraphDataset
from graphxai.datasets.feature import make_structured_feature
from graphxai.datasets.utils.bound_graph import build_bound_graph

from graphxai.datasets.utils.feature_generators import gaussian_lv_generator
from graphxai.datasets.utils.label_generators import bound_graph_label

class ShapeGraph(NodeDataset):
    '''
    BA Shapes dataset with keyword arguments for different planting, 
        insertion, labeling, and feature generation methods

    ..note:: Flag and circle shapes not yet implemented

    TODO: NEED TO FIX DOCSTRING

    Args:
        num_hops (int): Number of hops for each node's enclosing 
            subgraph. Should correspond to number of graph convolutional
            layers in GNN. 
        n (int): For global planting method, corresponds to the total number of 
            nodes in graph. If using local planting method, corresponds to the 
            starting number of nodes in graph.
        m (int): Number of edges per node in graph.
        num_shapes (int): Number of shapes for given planting strategy.
            If planting strategy is global, total number of shapes in
            the graph. If planting strategy is local, number of shapes
            per num_hops - hop neighborhood of each node.
        model_layers (int, optional): Number of layers within the GNN that will
            be explained. This defines the extent of the ground-truth explanations
            that are created by the method. (:default: :obj:`3`)
        shape (str, optional): Type of shape to be inserted into graph.
            Options are `'house'`, `'flag'`, `'circle'`, and `'multiple'`. 
            If `'multiple'`, random shapes are generated to insert into 
            the graph.
            (:default: :obj:`'house'`)
        graph_construct_strategy (str, optional): Type of insertion strategy for 
            each motif. Options are `'plant'` or `'staple'`.
            (:default: :obj:`'plant'`)
        shape_insert_strategy (str, optional): How to decide where shapes are 
            planted. 'global' method chooses random nodes from entire 
            graph. 'local' method enforces a lower bound on number of 
            shapes in the (num_hops)-hop neighborhood of each node. 
            'neighborhood upper bound' enforces an upper-bound on the 
            number of shapes per num_hops-hop neighborhood. 'bound_12' 
            enforces that all nodes in the graph are either in 1 or 2
            houses. 
            (:default: :obj:`'global'`)
        feature_method (str, optional): How to generate node features.
            Options are `'network stats'` (features are network statistics),
            `'gaussian'` (features are random gaussians), 
            `'gaussian_lv` (gaussian latent variables), and 
            `'onehot'` (features are random one-hot vectors) 
            (:default: :obj:`'network stats'`)
        labeling_method (str, optional): Rule of how to label the nodes.
            Options are `'feature'` (only label based on feature attributes), 
            `'edge` (only label based on edge structure), 
            `'edge and feature'` (label based on both feature attributes
            and edge structure). (:default: :obj:`'edge and feature'`)

        kwargs: Additional arguments
            shape_upper_bound (int, Optional): Number of maximum shapes
                to add per num_hops-hop neighborhood in the 'neighborhood
                upper bound' planting policy.

    Members:
        G (nx.Graph): Networkx version of the graph for the dataset
            - Contains many values per-node: 
                1. 'shape': which motif a given node is within
                2. 'shapes_in_khop': number of motifs within a (num_hops)-
                    hop neighborhood of the given node.
                    - Note: only for bound graph
    '''

    def __init__(self,  
        model_layers: int = 3,
        shape: Union[str, nx.Graph] = 'house',
        seed: Optional[int] = None):

        super().__init__(name = 'ShapeGraph', num_hops = model_layers)

        self.in_shape = []
        self.graph = None
        self.model_layers = model_layers

        # Barabasi-Albert parameters (will generalize later)
        self.seed = seed

        # Get shape:
        self.shape_method = ''
        if isinstance(shape, nx.Graph):
            self.insert_shape = shape
        else:
            self.insert_shape = None
            shape = shape.lower()
            self.shape_method = shape
            if shape == 'house':
                self.insert_shape = house
            elif shape == 'flag':
                pass
            elif shape == 'circle':
                self.insert_shape = pentagon # 5-member ring

            if shape == 'random':
                # random_shape imported from utils.shapes
                self.get_shape = partial(random_shape, n=3) 
                # Currenlty only support for 3-member shape bank (n=3)
            else:
                self.get_shape = lambda: (self.insert_shape, 1)

            assert shape != 'random', 'Multiple shapes not yet supported for bounded graph'

        #self.y_before_x = (self.feature_method == 'gaussian_lv')

        # Build graph:
        self.G = build_bound_graph(
                shape = self.insert_shape, 
                num_subgraphs = 10, 
                inter_sg_connections = 1,
                prob_connection = 1,
                num_hops = 1,
                base_graph = 'ba',
                )

        self.generate_shape_graph() # Performs planting, augmenting, etc.
        self.num_nodes = self.G.number_of_nodes() # Number of nodes in graph

        # Set random splits for size n graph:
        range_set = list(range(self.num_nodes))
        random.seed(1234) # Seed random before making splits
        train_nodes = random.sample(range_set, int(self.num_nodes * 0.7))
        test_nodes  = random.sample(range_set, int(self.num_nodes * 0.25))
        valid_nodes = random.sample(range_set, int(self.num_nodes * 0.05))

        self.fixed_train_mask = torch.tensor([s in train_nodes for s in range_set], dtype=torch.bool)
        self.fixed_test_mask = torch.tensor([s in test_nodes for s in range_set], dtype=torch.bool)
        self.fixed_valid_mask = torch.tensor([s in valid_nodes for s in range_set], dtype=torch.bool)

    def generate_shape_graph(self):
        '''
        Generates the full graph with the given insertion and planting policies.

        :rtype: :obj:`torch_geometric.Data`
        Returns:
            data (torch_geometric.Data): Entire generated graph.
        '''

        gen_labels = bound_graph_label(self.G)
        y = torch.tensor([gen_labels(i) for i in self.G.nodes], dtype=torch.long)
        self.yvals = y.detach().clone() # MUST COPY TO AVOID MAJOR BUGS

        gen_features, self.feature_imp_true = gaussian_lv_generator(self.G, self.yvals, seed = self.seed)
        x = torch.stack([gen_features(i) for i in self.G.nodes]).float()

        for i in self.G.nodes:
            self.G.nodes[i]['x'] = gen_features(i)

        edge_index = to_undirected(torch.tensor(list(self.G.edges), dtype=torch.long).t().contiguous())

        self.graph = Data(
            x=x, 
            y=y,
            edge_index = edge_index, 
            shape = torch.tensor(list(nx.get_node_attributes(self.G, 'shape').values()))
        )

        # Generate explanations:
        self.explanations = [self.explanation_generator(n) for n in self.G.nodes]

    def explanation_generator(self, node_idx):

        # Label node and edge imp based off of each node's proximity to a house

        # Find nodes in num_hops
        original_in_num_hop = set([self.G.nodes[n]['shape'] for n in khop_subgraph_nx(node_idx, 1, self.G) if self.G.nodes[n]['shape'] != 0])

        # Tag all nodes in houses in the neighborhood:
        khop_nodes = khop_subgraph_nx(node_idx, self.model_layers, self.G)
        node_imp_map = {i:(self.G.nodes[i]['shape'] in original_in_num_hop) for i in khop_nodes}
            # Make map between node importance in networkx and in pytorch data

        khop_info = k_hop_subgraph(
            node_idx,
            num_hops = self.model_layers,
            edge_index = to_undirected(self.graph.edge_index)
        )

        node_imp = torch.tensor([node_imp_map[i.item()] for i in khop_info[0]], dtype=torch.double)

        # Get edge importance based on edges between any two nodes in motif
        in_motif = khop_info[0][node_imp.bool()] # Get nodes in the motif
        edge_imp = torch.zeros(khop_info[1].shape[1], dtype=torch.double)
        for i in range(khop_info[1].shape[1]):
            if khop_info[1][0,i] in in_motif and khop_info[1][1,i] in in_motif:
                edge_imp[i] = 1

        exp = Explanation(
            feature_imp=self.feature_imp_true,
            node_imp = node_imp,
            edge_imp = edge_imp,
            node_idx = node_idx
        )

        exp.set_enclosing_subgraph(khop_info)
        return exp


    def visualize(self, shape_label = False):
        '''
        Args:
            shape_label (bool, optional): If `True`, labels each node according to whether
            it is a member of an inserted motif or not. If `False`, labels each node 
            according to its y-value. (:default: :obj:`True`)
        '''

        Gitems = list(self.G.nodes.items())
        node_map = {Gitems[i][0]:i for i in range(self.G.number_of_nodes())}

        if shape_label:
            y = [int(self.G.nodes[i]['shape'] > 0) for i in range(self.num_nodes)]
        else:
            ylist = self.graph.y.tolist()
            y = [ylist[node_map[i]] for i in self.G.nodes]

        node_weights = {i:node_map[i] for i in self.G.nodes}

        pos = nx.kamada_kawai_layout(self.G)
        _, ax = plt.subplots()
        nx.draw(self.G, pos, node_color = y, labels = node_weights, ax=ax)
        ax.set_title('BA Houses')
        plt.tight_layout()
        plt.show()
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
from .utils.shapes import *
from .dataset import NodeDataset, GraphDataset
from graphxai.datasets.feature import make_structured_feature
from graphxai.datasets.utils.bound_graph import build_bound_graph

from .utils.feature_generators import net_stats_generator, random_continuous_generator
from .utils.feature_generators import random_onehot_generator, gaussian_lv_generator
from .utils.label_generators import motif_id_label, binary_feature_label, number_motif_equal_label
from .utils.label_generators import bound_graph_label, logical_edge_feature_label

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
        num_hops: int, 
        n: int, 
        m: int, 
        num_shapes: int, 
        model_layers: int = 3,
        shape: Union[str, nx.Graph] = 'house',
        seed: Optional[int] = None,
        graph_construct_strategy: Optional[str] = 'plant',
        shape_insert_strategy: Optional[str] = 'bound_12',
        feature_method: Optional[str] = 'gaussian_lv',
        labeling_method: Optional[str] = 'edge',
        **kwargs):

        super().__init__(name = 'ShapeGraph', num_hops = num_hops)

        self.in_shape = []
        self.graph = None
        self.num_shapes = num_shapes
        self.shape_insert_strategy = shape_insert_strategy
        self.graph_construct_strategy = graph_construct_strategy
        self.model_layers = model_layers

        # Barabasi-Albert parameters (will generalize later)
        self.n = n
        self.m = m
        self.seed = seed

        self.feature_method = feature_method.lower()
        self.labeling_method = labeling_method.lower()
        if self.labeling_method == 'feature and edge':
            self.labeling_method = 'edge and feature'

        if shape_insert_strategy == 'neighborhood upper bound' and num_shapes is None:
            num_shapes = n # Set to maximum if num_houses left at None

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

            if shape_insert_strategy.lower() == 'bound_12':
                assert shape != 'random', 'Multiple shapes not yet supported for bounded graph'

        self.y_before_x = (self.feature_method == 'gaussian_lv')

        if self.shape_insert_strategy == 'neighborhood upper bound':
            assert 'shape_upper_bound' in kwargs
            self.upper_bound = kwargs['shape_upper_bound']

        # Build graph:
        self.init_graph() # Builds
        self.generate_shape_graph() # Performs planting, augmenting, etc.
        self.num_nodes = self.G.number_of_nodes() # Number of nodes in graph

        # Set random splits for size n graph:
        range_set = list(range(self.num_nodes))
        train_nodes = random.sample(range_set, int(self.num_nodes * 0.7))
        test_nodes  = random.sample(range_set, int(self.num_nodes * 0.25))
        valid_nodes = random.sample(range_set, int(self.num_nodes * 0.05))

        self.fixed_train_mask = torch.tensor([s in train_nodes for s in range_set], dtype=torch.bool)
        self.fixed_test_mask = torch.tensor([s in test_nodes for s in range_set], dtype=torch.bool)
        self.fixed_valid_mask = torch.tensor([s in valid_nodes for s in range_set], dtype=torch.bool)

    def init_graph(self):
        '''
        Returns a Barabasi-Albert graph with desired parameters
        '''
        if self.shape_insert_strategy == 'bound_12':
            self.G = build_bound_graph(
                shape = self.insert_shape, 
                num_subgraphs = 10, 
                inter_sg_connections = 1,
                prob_connection = 1,
                num_hops = self.num_hops,
                base_graph = 'ba',
                )
        else:
            self.G = nx.barabasi_albert_graph(self.n, self.m, seed = self.seed)

    def generate_shape_graph(self):
        '''
        Generates the full graph with the given insertion and planting policies.

        :rtype: :obj:`torch_geometric.Data`
        Returns:
            data (torch_geometric.Data): Entire generated graph.
        '''

        if self.shape_insert_strategy != 'bound_12':
            nx.set_node_attributes(self.G, 0, 'shape')
            nx.set_node_attributes(self.G, 0, 'motif_id')

        shape_insertion = self.staple if self.graph_construct_strategy == 'staple' else self.plant

        in_shape = set()
        shape_keys = {}

        if self.shape_insert_strategy == 'local':
            beginning_nodes = list(self.G.nodes)
            running_shape_code = 1
            for n in beginning_nodes:
                # Get shape from generator or static insert
                for i in range(self.num_shapes):
                    if self.insertion_shape is None:
                        shape, motif_id = self.get_shape()
                        #shape_keys[i] = shape_key # Add shape key to dict
                    else:
                        shape = self.insertion_shape.copy()
                        motif_id = 0 # Only have one shape

                    #shape = self.insertion_shape.copy() if self.insertion_shape is not None else self.get_shape()
                    in_shape_new = shape_insertion( 
                            in_shape = in_shape.copy(), 
                            shape = shape, 
                            shape_code = running_shape_code,
                            node_idx = n,
                            motif_id = motif_id
                        )

                    if len(in_shape_new) == len(in_shape):
                        # Break if we don't add anything
                        break

                    running_shape_code += 1
                    in_shape = in_shape_new

        elif self.shape_insert_strategy == 'global':
            for i in range(self.num_shapes):
                if self.insertion_shape is None:
                    shape, motif_id = self.get_shape()
                    #shape_keys[i] = shape_key # Add shape key to dict
                else:
                    shape = self.insertion_shape.copy()
                    motif_id = 0 # Only have one shape

                #shape = self.insertion_shape.copy() if self.insertion_shape is not None else self.get_shape()
                in_shape = shape_insertion( 
                        in_shape = in_shape, 
                        shape = shape, 
                        shape_code = i + 1,
                        motif_id = motif_id
                    )
                running_shape_code = i + 1

        elif self.shape_insert_strategy == 'neighborhood upper bound':
            nx.set_node_attributes(self.G, 0, 'shapes_nearby')
            for i in range(self.num_shapes):
                if self.insertion_shape is None:
                    shape, motif_id = self.get_shape()
                else:
                    shape = self.insertion_shape.copy()
                    motif_id = 0 # Only have one shape
                #shape = self.insertion_shape.copy() if self.insertion_shape is not None else self.get_shape()
                in_shape = shape_insertion( 
                        in_shape = in_shape, 
                        shape = shape, 
                        shape_code = i + 1,
                        motif_id = motif_id
                    )
                if in_shape is None: # Returned after we've inserted all the shapes needed
                    break
                running_shape_code = i + 1

        # If using bound_12, skip building graph - it's already built by upstream class

        if self.shape_insert_strategy != 'bound_12':
            self.shapes_in_graph = running_shape_code #Can later get number of shapes in the whole graph

        if self.y_before_x:
            gen_labels = self.labeling_rule()
            y = torch.tensor([gen_labels(i) for i in self.G.nodes], dtype=torch.long)
            self.yvals = y.detach().clone() # MUST COPY TO AVOID MAJOR BUGS

        gen_features = self.feature_generator()
        x = torch.stack([gen_features(i) for i in self.G.nodes]).float()

        for i in self.G.nodes:
            self.G.nodes[i]['x'] = gen_features(i)

        if not self.y_before_x:
            gen_labels = self.labeling_rule()
            y = torch.tensor([gen_labels(i) for i in self.G.nodes], dtype=torch.long)

        edge_index = to_undirected(torch.tensor(list(self.G.edges), dtype=torch.long).t().contiguous())

        self.graph = Data(
            x=x, 
            y=y,
            edge_index = edge_index, 
            shape = torch.tensor(list(nx.get_node_attributes(self.G, 'shape').values()))
        )

        # Generate explanations
        exp_gen = self.explanation_generator()

        self.explanations = [exp_gen(n) for n in self.G.nodes]

    def plant(self, 
            in_shape: set,
            shape: nx.Graph, 
            shape_code: int,
            node_idx: int = None,
            motif_id: int = None
        ):
        '''
        Plants a shape given that shape's nodes and edges
        Args:
            G (nx.Graph): 
            in_shape (set): Graph nodes that are already contained in a shape.
            shape (nx.Graph): Shape to plant in graph
            shape_code (int): Code by which to associate the shape
            node_idx (int): Node index for which to plant the shape around
            motif_id (int, optional): Unique identifier for type of shape. Will
                be added as a node attribute to the graph.
        '''

        if self.shape_insert_strategy == 'local' and node_idx is None:
            raise AttributeError("node_idx argument must be an int if shape_insert_strategy == 'local'")

        shape = shape.copy() # Ensure no corruption across function calls

        if self.shape_insert_strategy == 'neighborhood upper bound':
            pivot_set = set(self.G.nodes) - in_shape
            number_nodes = len(pivot_set)
            for _ in range(number_nodes):
                pivot = self.__random_node(sample_nodes = pivot_set)
                pivot_set.remove(pivot)
                if self.__check_neighborhood_bound_validity(pivot):
                    break
            else:
                return None # Didn't find a node that matched condition

            # Find all nodes in khop and uptick their nearby_count
            self.__increment_nearby_shapes(pivot)
                
        else:
            pivot = self.__random_node(in_shape, node_idx) # Choose random node based on strategy
        if pivot is None:
            return in_shape

        # Choose random node in shape:
        to_pivot = random.choice(list(shape.nodes))
        convert = {to_pivot: pivot} # Conversion dictionary for nodes in shape

        mx_nodes = max(list(self.G.nodes))

        i = 1
        for n in shape.nodes:
            if n == to_pivot:
                continue
            convert[n] = mx_nodes + i # Unique node label not in graph
            i += 1

        # Define new shape based on node labels
        new_shape = nx.relabel.relabel_nodes(shape, convert)

        for n in new_shape.nodes: # Add all new house nodes
            in_shape.add(n)

        # Add nodes and edges into G:
        self.G.add_nodes_from(list(set(list(new_shape.nodes)) - set([pivot])))
        self.G.add_edges_from(list(new_shape.edges))

        for n in new_shape.nodes:
            self.G.nodes[n]['shape'] = shape_code # Assign code for shape
            self.G.nodes[n]['motif_id'] = motif_id

        return in_shape

    def staple(self, 
            in_shape: set,
            shape: nx.Graph, 
            node_idx = None, 
            method = 'global'
        ):

        if node_idx is None:
            raise AttributeError("node_idx argument must be an int if method == 'local'")

        pivot = self.__random_node(method, node_idx)

        raise NotImplementedError()

    def feature_generator(self):
        '''
        Returns function to generate features for one node_idx
        '''
        if self.feature_method == 'network stats':
            get_feature = net_stats_generator(self.G)

        elif self.feature_method == 'gaussian':
            get_feature = random_continuous_generator(3) # Gets length 3 vector

        elif self.feature_method == 'onehot':
            get_feature = random_onehot_generator(3) # Gets length 3 vector

        elif self.feature_method == 'gaussian_lv':
            get_feature, self.feature_imp_true = gaussian_lv_generator(self.G, self.yvals, seed = self.seed)
        
        else:
            raise NotImplementedError(f'{self.feature_method} feature method not supported')

        return get_feature

    def labeling_rule(self):
        '''
        Labeling rule for each node
        '''

        if self.labeling_method == 'edge':
            # Label based soley on edge structure
            # Based on number of houses in neighborhood
            if self.shape_method == 'random':
                get_label = motif_id_label(self.G, self.num_hops)
            else:
                if self.shape_insert_strategy == 'bound_12':
                    get_label = bound_graph_label(self.G)
                else:
                    get_label = number_motif_equal_label(self.G, self.num_hops, equal_number=1)

        elif self.labeling_method == 'feature':
            # Label based solely on node features
            # Based on if feature[1] > median of all nodes
            get_label = binary_feature_label(self.G)

        elif self.labeling_method == 'edge and feature':
            # Note: currently built to only work with network statistics features
            get_label = logical_edge_feature_label(self.G, num_hops = self.num_hops, feature_method = 'median')

        else:
            raise NotImplementedError() 
                
        return get_label

    def explanation_generator(self):

        # Label node and edge imp based off of each node's proximity to a house

        if self.labeling_method == 'edge':
            def exp_gen(node_idx):

                if self.feature_method == 'gaussian_lv':
                    # Set feature_imp to mask generated earlier:
                    feature_imp = self.feature_imp_true
                else:
                    raise NotImplementedError('Need to define node importance for other terms')

                # Find nodes in num_hops
                original_in_num_hop = set([self.G.nodes[n]['shape'] for n in khop_subgraph_nx(node_idx, self.num_hops, self.G) if self.G.nodes[n]['shape'] != 0])

                # Tag all nodes in houses in the neighborhood:
                khop_nodes = khop_subgraph_nx(node_idx, self.model_layers, self.G)
                #node_imp_map = {i:(self.G.nodes[i]['shape_number'] > 0) for i in khop_nodes}
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
                    feature_imp=feature_imp,
                    node_imp = node_imp,
                    edge_imp = edge_imp,
                    node_idx = node_idx
                )

                exp.set_enclosing_subgraph(khop_info)
                #exp.set_whole_graph(x = self.x, edge_index = self.graph.edge_index)
                return exp
        else:
            def exp_gen(node_idx):
                return None
        return exp_gen

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

    def __random_node(self, in_shape: set = None, node_idx = None, sample_nodes: set = None):
        '''
        Args:
            sample_nodes (set): Nodes to sample from within the graph. Use exclusively for
                'neighborhood upper bound' planting method.
        '''

        if self.shape_insert_strategy == 'local':
            nodes_in_khop = set(khop_subgraph_nx(node_idx, self.num_hops, self.G))
            #khop_in_shape = nodes_in_khop.intersection(in_shape)
            # Choose pivot node from all those not already in a shape:
            not_in_khop = list(nodes_in_khop - in_shape)
            pivot = random.choice(not_in_khop) \
                if len(nodes_in_khop.intersection(in_shape)) <= self.num_shapes else None

        elif self.shape_insert_strategy == 'global':
            pivot = random.choice(list(self.G.nodes))
            while pivot in in_shape:
                pivot = random.choice(list(self.G.nodes))

        elif self.shape_insert_strategy == 'neighborhood upper bound':
            pivot = random.choice(list(sample_nodes))

        return pivot

    def __check_neighborhood_bound_validity(self, node_idx):
        '''
        Checks if a node is a valid planting pivot for a shape within the 
            'neighborhood upper bound' method
        '''
        nodes_in_khop = khop_subgraph_nx(node_idx, self.num_hops, self.G)
        # num_unique_houses = len(np.unique([self.G.nodes[ni]['shape'] for ni in nodes_in_khop if self.G.nodes[ni]['shape'] > 0 ]))
        # if num_unique_houses < self.upper_bound:
        #     return False
        # Search for number of shapes in each node's k-hop:
        for n in nodes_in_khop:
            if self.G.nodes[n]['shapes_nearby'] >= self.upper_bound:
                return False

        return True

    def __increment_nearby_shapes(self, node_idx):
        '''
        Increments number of nearby shapes for all nodes in neighborhood of node_idx
        '''
        nodes_in_khop = khop_subgraph_nx(node_idx, self.num_hops, self.G)
        for n in nodes_in_khop: # INCLUDES node_idx
            self.G.nodes[n]['shapes_nearby'] += 1
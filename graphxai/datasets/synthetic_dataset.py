import torch
import random
import numpy as np
import networkx as nx
from typing import Optional

from torch_geometric.utils import to_networkx, from_networkx, to_undirected
from torch_geometric.data import Data
from .dataset import NodeDataset, GraphDataset
from graphxai.utils.nx_conversion import khop_subgraph_nx

class ShapeGraph(NodeDataset):
    '''
    Base class for all shape datasets. Datasets consist of a generated 
    graph that is then used to randomly plant motifs. Prediction tasks 
    and motif insertion methods are defined by inheriting classes.

    ..note:: `'staple'` method not yet implemented

    Args:
        name (str): Name of dataset.
        num_hops (int): Number of hops for each node's enclosing 
            subgraph. Should correspond to number of graph convolutional
            layers in GNN. 
        num_shapes (int): Number of shapes for given planting strategy.
            If planting strategy is global, total number of shapes in
            the graph. If planting strategy is local, number of shapes
            per num_hops - hop neighborhood of each node.
        graph_construct_strategy (str, optional): Type of insertion strategy for 
            each motif. Options are `'plant'` or `'staple'`.
            (:default: :obj:`'plant'`)
        shape_insert_strategy (str, optional): How to decide where shapes are 
            inserted. 'global' method chooses random nodes from entire 
            graph. 'local' method enforces a lower bound on number of 
            shapes in the (num_hops)-hop neighborhood of each node. 
            'neighborhood upper bound' enforces an upper-bound on the 
            number of shapes per num_hops-hop neighborhood.
            (:default: :obj:`'global'`)
        insert_shape (nx.Graph, optional): Shape to insert into graph. If `None`,
            `self.get_shape` function must be defined, will be called upon shape
            insertion. (:default: :obj:`None`)
        kwargs: Additional arguments
            shape_upper_bound (int, Optional): Number of maximum shapes
                to add per num_hops-hop neighborhood in the 'neighborhood
                upper bound' planting policy.

    Members:
        G (nx.Graph): Networkx graph for the entire dataset.
        graph (torch_geometric.data.Data): Entire graph for the dataset.
        explanations (list): Explanations for each node prediction.
    '''
    def __init__(self, 
            name: str, 
            num_hops: int,
            num_shapes: int,
            graph_construct_strategy: Optional[str] = 'plant',
            shape_insert_strategy: Optional[str] = 'global',
            insert_shape: Optional[nx.Graph] = None,
            y_before_x: Optional[bool] = False,
            **kwargs
        ):
        super().__init__(name = name, num_hops = num_hops)

        self.in_shape = []
        self.graph = None
        self.num_shapes = num_shapes
        self.shape_insert_strategy = shape_insert_strategy
        self.graph_construct_strategy = graph_construct_strategy
        self.y_before_x = y_before_x

        #self.insertion_shape = kwargs['insertion_shape'] if 'insertion_shape' in kwargs else None
        self.insertion_shape = insert_shape

        if self.shape_insert_strategy == 'neighborhood upper bound':
            assert 'shape_upper_bound' in kwargs
            self.upper_bound = kwargs['shape_upper_bound']

        self.init_graph()
        self.generate_shape_graph()
        self.num_nodes = len(list(self.G.nodes)) # Number of nodes in graph

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
        Initializes graph for the class
            - Should set self.G, a nx.Graph object corresponding to overall graph
        '''
        # Must be implemented by child class
        raise NotImplementedError()

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

    def get_shape(self):
        '''Method for getting a shape (potentially random)'''
        raise NotImplementedError()

    def feature_generator(self):
        '''
        Uses self.G to 

        :rtype: Callable[[int], torch.Tensor]
            Function returned from this method must take one integer as argument (node index) 
            and return a feature for that node
        '''
        raise NotImplementedError()

    def labeling_rule(self):
        '''
        Args:
        
        :rtype: Callable[[int], int]
        '''
        raise NotImplementedError()

    def explanation_generator(self):
        '''
        Should generate explanations for one node_idx

        :rtype: Callable[[int], Explanation]
        '''
        raise NotImplementedError()

import torch
import random
import networkx as nx
import matplotlib.pyplot as plt
from typing import Optional
from torch_geometric.utils import k_hop_subgraph, to_undirected
from torch_geometric.data import Data

from graphxai.utils.nx_conversion import khop_subgraph_nx
from graphxai import Explanation
from ..utils.shapes import house
from .dataset import NodeDataset
from graphxai.datasets.utils.bound_graph import build_bound_graph

from ..utils.feature_generators import gaussian_lv_generator
from ..utils.label_generators import bound_graph_label

class ShapeGraph(NodeDataset):
    '''
    BA Shapes dataset for development
        - Only supports Gaussian latent variable feature generation and
          bounded graph construction

    Args:
        num_hops (int): Number of hops for each node's enclosing 
            subgraph. Should correspond to number of graph convolutional
            layers in GNN. 
        num_subgraphs (int, optional): Roughly controls size of output graph
        graph_sparsity (float, optional): Controls how sparse the subgraphs
            are connected in the bound graph. Should be float between 0 and 1.
        
    '''

    def __init__(self, 
        num_hops: int, 
        #seed: Optional[int] = None,
        num_subgraphs: Optional[int] = 10,
        graph_sparsity: Optional[float] = 1):

        super().__init__(name = 'ShapeGraph', num_hops = num_hops)

        self.in_shape = []

        # Build graph:
        self.G = build_bound_graph(
                    shape = house, 
                    num_subgraphs = num_subgraphs, 
                    inter_sg_connections = 1,
                    prob_connection = graph_sparsity,
                    num_hops = self.num_hops,
                    base_graph = 'ba',
                )
        self.generate_shape_graph() # Makes explanations, 


        # Getting masks:
        self.num_nodes = self.G.number_of_nodes() # Number of nodes in graph

        # Set random splits for size n graph:
        range_set = list(range(self.num_nodes))
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

        nx.set_node_attributes(self.G, 0, 'shape')
        nx.set_node_attributes(self.G, 0, 'motif_id')

        # Set y before x:
        gen_labels = bound_graph_label(self.G)
        y = torch.tensor([gen_labels(i) for i in self.G.nodes], dtype=torch.long)

        gen_features, self.feature_imp_true = gaussian_lv_generator(self.G, y.detach().clone(), seed = self.seed)
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

        # Generate explanations
        exp_gen = self.explanation_generator()

        self.explanations = [exp_gen(n) for n in self.G.nodes]


    def explanation_generator(self):

        # Label node and edge imp based off of each node's proximity to a house

        def exp_gen(node_idx):

            # Set feature_imp to mask generated earlier:
            feature_imp = self.feature_imp_true

            # Tag all nodes in houses in the neighborhood:
            khop_nodes = khop_subgraph_nx(node_idx, self.num_hops, self.G)
            node_imp_map = {i:(self.G.nodes[i]['shape_number'] > 0) for i in khop_nodes}
                # Make map between node importance in networkx and in pytorch data

            khop_info = k_hop_subgraph(
                node_idx,
                num_hops = self.num_hops,
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
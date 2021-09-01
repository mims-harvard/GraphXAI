import torch
import random
import numpy as np
import networkx as nx

from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

from sklearn.model_selection import train_test_split

from .dataset import NodeDataset
from graphxai.utils import Explanation

class BAHouses(NodeDataset):

    def __init__(self, num_hops, n, m, num_houses, seed = 1234):
        super().__init__(name='BAHouses', num_hops=num_hops)
        self.n = n
        self.m = m
        self.seed = seed

        # Generate data:
        self.graph = self.new_data(num_houses)

        # Formulate static split (independent of seed):
        # Keep seed constant for reproducible splits
        train_mask, rem_mask = train_test_split(list(range(self.n)), 
                                test_size = 0.3, 
                                random_state = 1234)

        valid_mask, test_mask = train_test_split(rem_mask, 
                                test_size = 1.0 / 3,
                                random_state = 5678)

        self.static_train_mask = torch.tensor([i in train_mask for i in range(n)], dtype = torch.bool)
        self.static_valid_mask = torch.tensor([i in valid_mask for i in range(n)], dtype = torch.bool)
        self.static_test_mask  = torch.tensor([i in test_mask  for i in range(n)], dtype = torch.bool)

        self.graph = self.new_data(num_houses)
        self.__make_explanations() # Sets self.explanation

    def new_data(self, num_houses):
        '''
        Generates new data for the class
        Resets all within-class graph components
        '''
        self.in_house = set()
        random.seed(self.seed)
        BAG = self.__make_BA_shapes(num_houses)
        # For now, making all data with multiple features:
        data = self.__make_data(BAG, True)
        random.seed() # Seed back to random
        return data

    def __make_explanations(self):
        self.explanations = []
        for i in range(self.n):
            exp = Explanation()
            enc_subgraph = self.get_enclosing_subgraph(node_idx=i)
            exp.set_enclosing_subgraph(enc_subgraph)
            # Whole graph left empty on exp (redundant)
            exp.node_imp = self.graph.y[enc_subgraph.nodes] # Set to y values
            exp.node_idx = i
            self.explanations.append(exp)

    def __make_BA_shapes(self, num_houses = 1, make_pyg = False):
        start_n = self.n - (4 * num_houses)
        G = nx.barabasi_albert_graph(start_n, self.m, seed = self.seed)
        self.node_attr = torch.zeros(self.n, dtype = torch.long)

        # Set num_houses:
        self.num_houses = num_houses

        for ec in range(1, num_houses + 1):
            G = self.__make_house(G, ec)

        G = G.to_directed()

        return G

    def __make_data(self, G, multiple_features = False):
        # One hot lookup:
        onehot = {}
        for i in range(self.num_houses + 1):
            onehot[i] = [0] * (self.num_houses + 1)
            onehot[i][i] = 1

        # Encode with degree as feature
        deg_cent = nx.degree_centrality(G)
        x = torch.stack(
                [torch.tensor([G.degree[i], nx.clustering(G, i), deg_cent[i]]) 
                for i in range(len(list(G.nodes)))]
            ).float()

        edge_index = torch.tensor(list(G.edges), dtype=torch.long)

        y = [1 if i in self.in_house else 0 for i in range(self.n)]

        data = Data(x=x, y = torch.tensor(y, dtype = torch.long), edge_index=edge_index.t().contiguous())

        return data

    def __make_house(self, G, encode_num = 1):
        pivot = random.choice(list(set(G.nodes) - self.in_house))
        mx = np.max(G.nodes)
        new_nodes = [mx + i for i in range(1, 5)]
        house_option = random.choice(list(range(3)))
        G.add_nodes_from(new_nodes)

        if house_option == 0:
            connections = [(new_nodes[i], new_nodes[i+1]) for i in range(3)]
            connections += [(new_nodes[-1], new_nodes[1])]
            G.add_edges_from(connections)
            G.add_edge(pivot, new_nodes[0])
            G.add_edge(pivot, new_nodes[-1])

        elif house_option == 1:
            connections = [(new_nodes[i], new_nodes[i+1]) for i in range(3)]
            connections += [(new_nodes[0], new_nodes[-1])]
            G.add_edges_from(connections)
            G.add_edge(pivot, new_nodes[0])
            G.add_edge(pivot, new_nodes[-1])

        elif house_option == 2:
            connections = [(new_nodes[i], new_nodes[i+1]) for i in range(3)]
            G.add_edges_from(connections)
            G.add_edge(pivot, new_nodes[0])
            G.add_edge(pivot, new_nodes[2])
            G.add_edge(pivot, new_nodes[-1])

        house = new_nodes + [pivot]
        for n in house:
            self.node_attr[n] = encode_num # Encoding number for final node attributes
            self.in_house.add(n) # Add to house tracker

        return G


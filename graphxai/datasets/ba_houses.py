import torch
import random
import numpy as np
import networkx as nx
from typing import Optional
import matplotlib.pyplot as plt

from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

from sklearn.model_selection import train_test_split

from graphxai.datasets.dataset import NodeDataset
from graphxai.utils import Explanation, node_mask_from_edge_mask


def make_edge_imp(subg_edge_index, subg_nodes, node_imp):
    '''
    Make edge importance assuming that source node is in a shape
    '''

    # Positive nodes:
    pos_nodes = set([s.item() for s in subg_nodes[(node_imp == 1)]])

    edge_mask = torch.zeros(subg_edge_index.shape[1])

    # Find all occurences of pos_nodes in subg_edge_index:
    for i in range(subg_edge_index.shape[1]):
        if (subg_edge_index[0,i].item() in pos_nodes) \
            and (subg_edge_index[1,i].item() in pos_nodes):
            edge_mask[i] = 1

    return edge_mask


class BAHouses(NodeDataset):
    '''
    Args:
        model_layers (int):
        n (int): 
        m (int):
        num_houses (int):
        seed (int): Seed for random generation of graph
    '''

    def __init__(self, 
        model_layers, 
        n: int, 
        m: int, 
        num_houses: int, 
        seed: Optional[int] = 1234):
        
        super().__init__(name='BAHouses', num_hops=model_layers)
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

        test_mask, valid_mask = train_test_split(rem_mask, 
                                test_size = 1.0 / 3,
                                random_state = 5678)

        self.fixed_train_mask = torch.tensor([i in train_mask for i in range(n)], dtype = torch.bool)
        self.fixed_valid_mask = torch.tensor([i in valid_mask for i in range(n)], dtype = torch.bool)
        self.fixed_test_mask  = torch.tensor([i in test_mask  for i in range(n)], dtype = torch.bool)

        self.graph = self.new_data(num_houses)
        self.__generate_explanations() # Sets self.explanation

    def new_data(self, num_houses):
        '''
        Generates new data for the class
        Resets all within-class graph components
        '''
        self.in_house = set()
        random.seed(self.seed)
        self.G = self.__generate_BA_shapes(num_houses)
        # For now, making all data with multiple features:
        data = self.__generate_data(self.G, True)
        random.seed() # Seed back to random
        return data

    def __generate_explanations(self):
        self.explanations = []
        for i in range(self.n):
            exp = Explanation()
            enc_subgraph = self.get_enclosing_subgraph(node_idx=i)
            exp.set_enclosing_subgraph(enc_subgraph)
            # Whole graph left empty on exp (redundant)
            exp.node_imp = self.graph.y[enc_subgraph.nodes] # Set to y values
            exp.node_idx = i
            # Edge imp is any
            exp.edge_imp = make_edge_imp(enc_subgraph.edge_index, enc_subgraph.nodes, exp.node_imp)
            exp.feature_imp = torch.ones(1, 1).float()
            self.explanations.append([exp]) #Must be a list

    def __generate_BA_shapes(self, num_houses = 1, make_pyg = False):
        start_n = self.n - (4 * num_houses)
        G = nx.barabasi_albert_graph(start_n, self.m, seed = self.seed)
        self.node_attr = torch.zeros(self.n, dtype = torch.long)

        # Set num_houses:
        self.num_houses = num_houses

        for ec in range(1, num_houses + 1):
            G = self.__plant_house(G, ec)

        G = G.to_directed()

        return G

    def __generate_data(self, G, multiple_features = False):
        # One hot lookup:
        onehot = {}
        for i in range(self.num_houses + 1):
            onehot[i] = [0] * (self.num_houses + 1)
            onehot[i][i] = 1

        # Encode with degree as feature
        deg_cent = nx.degree_centrality(G)
        # x = torch.stack(
        #         [torch.tensor([G.degree[i], nx.clustering(G, i), deg_cent[i]]) 
        #         for i in range(len(list(G.nodes)))]
        #     ).float()

        x = torch.ones(len(list(G.nodes)), 1).float()

        edge_index = torch.tensor(list(G.edges), dtype=torch.long)

        y = [1 if i in self.in_house else 0 for i in range(self.n)]

        data = Data(x=x, y = torch.tensor(y, dtype = torch.long), edge_index=edge_index.t().contiguous())

        return data

    def __plant_house(self, G, encode_num = 1):
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
        
    def visualize(self, shape_label = False):
        '''
        Args:
            shape_label (bool, optional): If `True`, labels each node according to whether
            it is a member of an inserted motif or not. If `False`, labels each node 
            according to its y-value. (:default: :obj:`True`)
        '''

        Gitems = list(self.G.nodes.items())
        node_map = {Gitems[i][0]:i for i in range(self.G.number_of_nodes())}

        ylist = self.graph.y.tolist()
        y = [ylist[node_map[i]] for i in self.G.nodes]

        node_weights = {i:node_map[i] for i in self.G.nodes}

        pos = nx.kamada_kawai_layout(self.G)
        _, ax = plt.subplots()
        nx.draw(self.G, pos, node_color = y, labels = node_weights, ax=ax)
        ax.set_title('BA Houses')
        plt.tight_layout()
        plt.show()

import torch
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from torch_geometric.data import Data


class BA_Houses:

    def __init__(self, n, m, seed = None):
        self.n = n
        self.m = m
        self.in_house = set()
        self.seed = seed

    def get_data(self, num_houses, test_size = 0.25, multiple_features = False):
        random.seed(self.seed)
        BAG = self.make_BA_shapes(num_houses)
        data = self.make_data(BAG, test_size, multiple_features)
        inhouse = self.in_house
        random.seed()
        return data, list(inhouse)

    def make_house(self, G, encode_num = 1):
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

    def make_data(self, G, test_size = 0.25, multiple_features = False):
        # One hot lookup:
        onehot = {}
        for i in range(self.num_houses + 1):
            onehot[i] = [0] * (self.num_houses + 1)
            onehot[i][i] = 1

        # Encode with degree as feature
        deg_cent = nx.degree_centrality(G)
        if multiple_features:
            x = torch.stack([torch.tensor([G.degree[i], nx.clustering(G, i), deg_cent[i]]) for i in range(len(list(G.nodes)))]).float()
        else:
            x = torch.stack([torch.tensor([G.degree[i]]) for i in range(len(list(G.nodes)))]).float()

        edge_index = torch.tensor(list(G.edges), dtype=torch.long)

        # Split out train and test:
        train_mask = torch.full((self.n,), False)
        test_mask = torch.full((self.n,), False)

        test_set = set(random.sample(list(range(self.n)), int(test_size * self.n)))
        for i in range(self.n):
            if i in test_set:
                test_mask[i] = True
            else:
                train_mask[i] = True

        #y = [0 if self.node_attr[i].item() == 0 else 1 for i in range(self.node_attr.shape[0])]
        y = [1 if i in self.in_house else 0 for i in range(self.n)]

        data = Data(x=x, y = torch.tensor(y, dtype = torch.long), edge_index=edge_index.t().contiguous(),
                    train_mask = train_mask, test_mask = test_mask)

        return data

    def make_BA_shapes(self, num_houses = 1, make_pyg = False):
        start_n = self.n - (4 * num_houses)
        G = nx.barabasi_albert_graph(start_n, self.m, seed = self.seed)
        self.node_attr = torch.zeros(self.n, dtype = torch.long)

        # Set num_houses:
        self.num_houses = num_houses

        for ec in range(1, num_houses + 1):
            G = self.make_house(G, ec)

        G = G.to_directed()

        return G

    def draw(self, G):
        pos = nx.kamada_kawai_layout(G)
        nx.draw(G, pos, node_color = self.node_attr, node_size = 400,
                cmap = plt.cm.Blues, arrows = False)
        plt.show()

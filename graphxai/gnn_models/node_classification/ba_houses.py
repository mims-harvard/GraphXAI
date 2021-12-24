import torch
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from torch_geometric.data import Data


class BA_Houses:

    def __init__(self, n, m, seed=None):
        self.n = n
        self.m = m
        self.in_house = set()
        self.seed = seed

    def get_data(self, num_houses, test_size=0.25):
        random.seed(self.seed)
        BAG = self.make_BA_shapes(num_houses)
        data = self.make_data(BAG, test_size)
        inhouse = self.in_house
        return data, list(inhouse)

    def make_house(self, G, encode_num=1):
        pivot = random.choice(list(set(G.nodes) - self.in_house))
        mx = np.max(G.nodes)
        new_nodes = [mx + i for i in range(1, 5)]
        house_option = random.choice(list(range(3)))
        G.add_nodes_from(new_nodes)

        if house_option == 0:
            edges = [(new_nodes[i], new_nodes[i+1]) for i in range(3)]
            edges += [(new_nodes[-1], new_nodes[1]), (pivot, new_nodes[0]), (pivot, new_nodes[-1])]
            G.add_edges_from(edges)
        elif house_option == 1:
            edges = [(new_nodes[i], new_nodes[i+1]) for i in range(3)]
            edges += [(new_nodes[0], new_nodes[-1]), (pivot, new_nodes[0]), (pivot, new_nodes[-1])]
            G.add_edges_from(edges)
        else:
            edges = [(new_nodes[i], new_nodes[i+1]) for i in range(3)]
            edges += [(pivot, new_nodes[0]), (pivot, new_nodes[2]), (pivot, new_nodes[-1])]
            G.add_edges_from(edges)

        nodes = new_nodes + [pivot]
        for node in nodes:
            self.node_attr[node] = encode_num  # Encoding number for final node attributes
            self.in_house.add(node)  # Add to house tracker

        return G, set(nodes), set(edges)

    def make_data(self, G, test_size=0.25):
        # One hot lookup:
        onehot = {}
        for i in range(self.num_houses + 1):
            onehot[i] = [0] * (self.num_houses + 1)
            onehot[i][i] = 1

        edge_index = torch.tensor(list(G.edges), dtype=torch.long)

        # Split out train and test:
        train_mask = torch.full((self.n,), False, dtype=bool)
        test_mask = torch.full((self.n,), False, dtype=bool)

        test_set = set(random.sample(list(range(self.n)), int(test_size * self.n)))
        for i in range(self.n):
            if i in test_set:
                test_mask[i] = True
            else:
                train_mask[i] = True

        y = torch.tensor([1 if i in self.in_house else 0 for i in range(self.n)],
                         dtype=torch.long)

        data = Data(y=y, edge_index=edge_index.t().contiguous(),
                    train_mask = train_mask, test_mask = test_mask)

        return data

    def make_BA_shapes(self, num_houses = 1):
        start_n = self.n - (4 * num_houses)
        G = nx.barabasi_albert_graph(start_n, self.m, seed = self.seed)
        self.node_attr = torch.zeros(self.n, dtype = torch.long)

        # Set num_houses
        self.num_houses = num_houses

        # Store all house nodes / edges
        self.houses = []

        for ec in range(1, num_houses + 1):
            G, house_nodes, house_edges = self.make_house(G, ec)
            self.houses.append((house_nodes, house_edges))

        G = G.to_directed()

        return G

    def draw(self, G):
        pos = nx.kamada_kawai_layout(G)
        nx.draw(G, pos, node_color = self.node_attr, node_size = 400,
                cmap = plt.cm.Blues, arrows = False)
        plt.show()

import sys
import torch
import random
import numpy as np
import networkx as nx
from typing import Optional
from networkx.algorithms.traversal.breadth_first_search import bfs_edges

from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from .dataset import NodeDataset
from graphxai.utils import Explanation

def plant_one_house(G, pivot, in_house, house_code):
    mx = np.max(list(G.nodes))
    new_nodes = [mx + i for i in range(1, 5)]
    house_option = random.choice(list(range(3)))

    G.add_nodes_from(new_nodes, house=house_code)
    # Set pivot's house code:
    G.nodes[pivot]['house'] = house_code

    connections = [(new_nodes[i], new_nodes[i+1]) for i in range(3)]

    if house_option == 0: # Pivot on bottom of house
        connections += [(new_nodes[-1], new_nodes[1])] # New node house connector
        connections += [(pivot, new_nodes[0]), (pivot, new_nodes[-1])]

    elif house_option == 1: # Pivot on top of roof
        connections += [(new_nodes[0], new_nodes[-1])]
        connections += [(pivot, new_nodes[0]), (pivot, new_nodes[-1])]

    elif house_option == 2: # Pivot on corner of roof
        connections += [(pivot, new_nodes[0]), (pivot, new_nodes[2]), (pivot, new_nodes[-1])]

    G.add_edges_from(connections)
    house = new_nodes + [pivot]
    for n in house:
        in_house.add(n) # Add to house tracker
        #G.nodes

    return G, in_house

def generate_explanations_BAHouses(data, node_idx, num_hops):

    khop_info = k_hop_subgraph(
        node_idx = int(node_idx), 
        num_hops = num_hops, 
        edge_index = data.edge_index)

    subgraph_nodes = khop_info[0]
    sub_edge_idx = khop_info[1]

    # Get class of node:
    house_code = data.house[node_idx]

    # All other nodes in neighborhood with this class:
    node_imp = torch.tensor([1 if data.house[i] == house_code else 0 for i in subgraph_nodes], dtype=torch.long)

    # Find all edges where both ends are in the appropriate house:
    edge_mask = torch.zeros(sub_edge_idx.shape[1], dtype=torch.long)
    house = data.house.detach()

    for i in range(sub_edge_idx.shape[1]):
        #Check both ends:
        e1, e2 = sub_edge_idx[0,i], sub_edge_idx[1,i]
        
        edge_mask[i] = ((house[e1] == house_code) and (house[e2] == house_code)).type(torch.long)

    exp = Explanation(
        node_imp = node_imp,
        edge_imp = edge_mask,
        node_idx = node_idx
    )
    exp.set_enclosing_subgraph(khop_info)
    exp.set_whole_graph(data.x, data.edge_index)

    return exp

def generate_BAHouses_graph_global(
        n: int, 
        m: int, 
        num_houses: int, 
        num_hops: int, 
        seed: int= None, 
        get_data: Optional[bool] = True,
        label_strategy: Optional[bool] = 0
    ):
    '''
    Generates a BAHouses graph with global planting method

    Args:
        n (int): Starting number of nodes in graph
        m (int): Number of connections to make per node
        num_houses (int): Number of total houses to plant
        num_hops (int): Number of hops in k-hop neighborhood of each node
            Should correspond to number of graph convolutional layers
            in the GNN.
        seed (int, optional): default
    '''
    #start_n = n - 4 * num_houses
    G = nx.barabasi_albert_graph(n, m, seed = seed)
    nx.set_node_attributes(G, 0, 'house')

    # Create all houses:
    in_house = set()
    house_code = 1
    for _ in range(num_houses):
        G, in_house = plant_house_global(G, in_house, house_code)
        house_code += 1

    G = G.to_directed()

    # Load x attributes:
    deg_cent = nx.degree_centrality(G)
    x = torch.stack(
                [torch.tensor([G.degree[i], nx.clustering(G, i), deg_cent[i]]) 
                for i in range(len(list(G.nodes)))]
            ).float()

    edge_index = torch.tensor(list(G.edges), dtype=torch.long)

    # Set node labels after planting:
    y = [1 if i in in_house else 0 for i in range(len(G.nodes))]

    if get_data:
        data = Data(
            x=x, 
            y = torch.tensor(y, dtype = torch.long), 
            edge_index=edge_index.t().contiguous()
        )

    else: # Mostly for debugging purposes
        return G, x, y, edge_index

    # Get explanations
    exp_list = []
    for node_idx in G.nodes:
        exp_list.append(generate_explanations_BAHouses(data, node_idx, num_hops))

    return data, exp_list

def plant_house_global(G, in_house, house_code):
    pivot = random.choice(list(set(G.nodes) - in_house))
    G, in_house = plant_one_house(G, pivot, in_house, house_code)
    return G, in_house

def generate_BAHouses_graph_local(
        n, 
        m, 
        k, 
        num_hops, 
        seed = None, 
        get_data = True, 
        in_hood_numbering = False,
        threshold = None,
        label_strategy: Optional[bool] = 0
    ):
    '''
    Args:
        n (int): starting number of nodes
    '''
    G = nx.barabasi_albert_graph(n, m, seed = seed)
    random.seed(seed)

    nx.set_node_attributes(G, 0, 'house')

    # Create all houses:
    in_house = set()
    running_house_code = 1
    for i in range(n):
        G, in_house, running_house_code = plant_house_local(G, 
            node_idx = i, 
            in_house=in_house, 
            k=k,
            num_hops = num_hops,
            running_house_code = running_house_code)

    G = G.to_directed()

    # Load x attributes:
    deg_cent = nx.degree_centrality(G)
    x = torch.stack(
                [torch.tensor([G.degree[i], nx.clustering(G, i), deg_cent[i]]) 
                for i in range(len(list(G.nodes)))]
            ).float()

    edge_index = torch.tensor(list(G.edges), dtype=torch.long)

    # Set node labels after planting:
    if label_strategy == 0:
        if in_hood_numbering:
            y = []
            for n in range(len(G.nodes)):
                khop_edges = nx.bfs_edges(G, n, depth_limit = num_hops)
                nodes_in_khop = set(np.unique(list(khop_edges))) - set([n])
                nodes_in_house = nodes_in_khop.intersection(in_house)
                num_unique_houses = len(np.unique([G.nodes[ni]['house'] for ni in nodes_in_house]))
                y.append(num_unique_houses)

            if threshold is not None:
                y = [1 if y[i] > threshold else 0 for i in range(len(y))]
            
        else:
            y = [1 if i in in_house else 0 for i in range(len(G.nodes))]

    elif label_strategy == 1:
        y = []
        for n in range(len(G.nodes)):
            khop_edges = nx.bfs_edges(G, n, depth_limit = num_hops)
            nodes_in_khop = set(np.unique(list(khop_edges))) - set([n])
            nodes_in_house = nodes_in_khop.intersection(in_house)
            num_unique_houses = len(np.unique([G.nodes[ni]['house'] for ni in nodes_in_house]))
            
            # Apply logic rule:
            if num_unique_houses == 1 and x[n,0] > 1:
                y.append(1)
            else:
                y.append(0)

    if get_data:
        data = Data(
            x=x, 
            y = torch.tensor(y, dtype = torch.long), 
            edge_index=edge_index.t().contiguous(),
            house = torch.tensor(list(nx.get_node_attributes(G, 'house').values()))
        )
        #print(nx.get_node_attributes(G, 'house'))
        #print(data.house)

    else: # Mostly for debugging purposes
        return G, x, y, edge_index

    exp_list = []
    #print(data)
    for node_idx in G.nodes:
        exp_list.append(generate_explanations_BAHouses(data, node_idx, num_hops))

    #exp = generate_explanations_BAHouses(data, node_idx, num_hops)

    return data, exp_list

def plant_house_local(
    G, 
    node_idx, 
    in_house, 
    k, 
    num_hops,
    running_house_code):

    edges = nx.bfs_edges(G, node_idx, depth_limit = num_hops)
    nodes_in_khop = set(np.unique(list(edges))) - set([node_idx])
    # Search for any in houses:
    khop_in_house = nodes_in_khop.intersection(in_house)

    # Get all unique houses in the neighborhood
    unique_houses_present = \
        np.unique([G.nodes[n]['house'] for n in khop_in_house])
    #print(unique_houses_present)

    # Subtract one to account for the zero-house
    to_fill = k - (len(unique_houses_present))
    #print(to_fill)
    
    if to_fill == 0:
        return G, in_house, running_house_code
    elif to_fill < 0:
        to_fill = 0
    
    not_in_house = nodes_in_khop - khop_in_house

    #print('len(not_in_house)', len(not_in_house))
    for _ in range(to_fill):
        if len(not_in_house) == 0:
            break
        pivot = random.choice(list(not_in_house))
        G, in_house = plant_one_house(G, pivot, in_house, 
            house_code = running_house_code)
        not_in_house = not_in_house - in_house

        # Increment to keep track of houses added:
        running_house_code += 1

    return G, in_house, running_house_code


if __name__ == '__main__':
    if sys.argv[1] == 'local':
        G, _, y, _, = generate_BAHouses_graph_local(
            n=50, 
            m = 1, 
            k = 1, 
            num_hops = 2,
            get_data = False,
            in_hood_numbering=True)
    elif sys.argv[1] == 'global':
        G, _, y, _, = generate_BAHouses_graph_global(
            n=50, 
            m = 1, 
            num_houses = 3, 
            num_hops = 2,
            get_data = False)
    
    # data, exp = generate_BAHouses_graph_local(
    #     n=50, 
    #     m = 1, 
    #     k = 1, 
    #     num_hops = 3,
    #     get_data = True)

    # Get nodes in houses, highlight by that:
    #node_weights = [G.nodes[n]['house'] for n in G.nodes]
    node_weights = y

    pos = nx.kamada_kawai_layout(G)
    nx.draw(G.to_undirected(), pos, node_color = node_weights)
    plt.show()
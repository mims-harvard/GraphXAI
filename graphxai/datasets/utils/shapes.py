import random
import torch
import networkx as nx

# def get_house():
#     '''
#     Defines house in terms of a nx graph.
#     '''
#     G = nx.Graph() # Different from G
#     nodes = list(range(5))
#     G.add_nodes_from(nodes)

#     # Define full cycle:
#     connections = [(nodes[i], nodes[i+1]) for i in nodes[:-1]]
#     connections += [(nodes[-1], nodes[0])]
#     connections += [(1, 4)] # Cross-house

#     G.add_edges_from(connections)
#     return G

def get_flag():
    pass

# Set common shapes:
house = nx.house_graph()
house_x = nx.house_x_graph()
diamond = nx.diamond_graph()
pentagon = nx.cycle_graph(n=5)
wheel = nx.wheel_graph(n=6)
star = nx.star_graph(n=5)
flag = None


def random_shape(n) -> nx.Graph:
    '''
    Outputs a random shape as nx.Graph

    ..note:: set `random.seed()` for seeding
    
    Args:
        n (int): Number of shapes in the bank to draw from
    
    '''
    shape_list = [
        house,
        pentagon,
        wheel
    ]
    i = random.choice(list(range(len(shape_list))))
    return shape_list[i], i + 1

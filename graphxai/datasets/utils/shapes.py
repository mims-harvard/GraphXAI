import networkx as nx

def get_house():
    '''
    Defines house in terms of a nx graph.
    '''
    G = nx.Graph() # Different from G
    nodes = list(range(5))
    G.add_nodes_from(nodes)

    # Define full cycle:
    connections = [(nodes[i], nodes[i+1]) for i in nodes[:-1]]
    connections += [(nodes[-1], nodes[0])]
    connections += [(1, 4)] # Cross-house

    G.add_edges_from(connections)
    return G

def get_flag():
    pass

# Set common shapes:
house = get_house()
flag = None

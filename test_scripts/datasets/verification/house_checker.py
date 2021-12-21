import time
import torch
import pandas as pd
import networkx as nx


from graphxai.datasets.shape_graph import ShapeGraph

# Script counts the number of houses in the graph by using a subgraph isomorphism counter

# Search parameters:
prob_connection_search = [0.01, 0.02, 0.025, 0.03, 0.04, 0.05, 0.1, 0.15, 0.5, 0.75, 1]
num_subgraph_search = [5, 10, 20, 30, 40, 50, 75, 100]

num_subgraphs = {subg: [] for subg in num_subgraph_search}
bad_shapes = {subg: [] for subg in num_subgraph_search}

for p in prob_connection_search:
    for sub in num_subgraph_search:
        print('prob connect:', p, 'Num_subg:', sub)
        start_time = time.time()
        #G = build_bound_graph(num_subgraphs = sub, num_hops = 1, prob_connection = p)
        bah = ShapeGraph(model_layers=3, num_subgraphs=sub, prob_connection=p, verify = False)
        G = bah.G
        # nx.draw(G)
        # plt.show()
        size_graph = bah.num_nodes
        t = time.time() - start_time
        print('\t Time:', t)
        print('\t Size:', size_graph)

        house = nx.house_graph()
        matcher = nx.algorithms.isomorphism.ISMAGS(graph = G, subgraph = house)
        i = 0
        bad = 0
        for iso in matcher.find_isomorphisms():
            i += 1

            # Check all nodes in isomorphism:
            nodes_found = iso.keys()
            shapes = [G.nodes[n]['shape'] for n in nodes_found]
            #print(shapes)

            if (sum([int(shapes[i] != shapes[i-1]) for i in range(1, len(shapes))]) > 0) \
                or (sum(shapes) == 0):
                bad += 1
                print(shapes)

        print('\t Matched Isomorphisms:', i)
        print('\t Difference:', sub - i)
        print('\t Bad shapes:', bad)

        bad_shapes[sub].append(bad)
        num_subgraphs[sub].append(i)
        
pd.DataFrame(num_subgraphs, index = prob_connection_search).to_csv('house_search.csv')
pd.DataFrame(bad_shapes, index = prob_connection_search).to_csv('bad_shapes.csv')
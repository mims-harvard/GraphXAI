import random
import numpy as np
import networkx as nx
from functools import partial
from collections import Counter
from .shapes import house
from graphxai.utils import khop_subgraph_nx
import matplotlib.pyplot as plt

from graphxai.gnn_models.node_classification.testing import *

def max_in_khop(G, node, num_hops, attr_measure: str):
    pass

def check_khop_bound(G, node, num_hops, attr_measure, bound, min_or_max = 1):
    '''
    Searches khop-neighborhood to ensure no nodes violate bound of a given
        measure
        Ex: ensure no nodes w/in k-hop have more than x # of adjacent shapes
    Args:
        bound (float): Bounding value for attribute (inclusive bound)
    '''

    subgraph_nodes = khop_subgraph_nx(node_idx=node, num_hops = num_hops, G = G)

    measures = [G.nodes[s][attr_measure] for s in subgraph_nodes]
    return (max(measures) <= bound) if min_or_max else (min(measures) >= bound)


def check_graph_bound(G, attr_measure, bound, min_or_max = 1):
    '''
    Searches an entire graph to ensure that some attribute does not violate a bound
    '''
    attrs = [G.nodes[n][attr_measure] for n in G.nodes]
    return (max(attrs) <= bound) if min_or_max else (min(attrs) >= bound)

def increment_shape_counter(G, nodes, attr_measure, exclude_nodes = None):
#def increment_shape_counter(G, insert_at_node, attr_measure, num_hops, exclude_nodes = None):
    '''
    Increments some attribute based on how man
    '''

    # subgraph_nodes = khop_subgraph_nx(node_idx = insert_at_node, num_hops = num_hops, G = G)

    # if exclude_nodes is not None:
    #     subgraph_nodes = list(set(subgraph_nodes) - set(exclude_nodes)) # Remove nodes from consideration

    # for n in subgraph_nodes:
    #     G.nodes[n][attr_measure] += 1

    for n in nodes:
        if exclude_nodes is not None:
            if n in exclude_nodes:
                continue
        G.nodes[n][attr_measure] += 1

    return G

def incr_on_unique_houses(nodes_to_search, G, num_hops, attr_measure, lower_bound, upper_bound):
    for n in nodes_to_search:
        khop = khop_subgraph_nx(node_idx = n, num_hops = num_hops, G = G)

        num_unique = torch.unique([G.nodes[i]['shape_number'] for i in khop]).shape[0]

        if num_unique < lower_bound or num_unique > upper_bound:
            return None
        else:
            G.nodes[n][attr_measure] = num_unique

    return G



def build_bound_graph(
        shape = house, 
        num_subgraphs = 5, 
        inter_sg_connections = 1,
        prob_connection = 0.5,
        num_hops = 2,
        base_graph = 'ba',
        ):
    # Create graph:
    if base_graph == 'ba':
        subgraph_generator = partial(nx.barabasi_albert_graph, n=5 * num_hops, m=1)

    subgraphs = []
    shape_node_per_subgraph = []
    original_shapes = []
    floor_counter = 0
    shape_number = 1
    for i in range(num_subgraphs):
        current_shape = shape.copy()
        nx.set_node_attributes(current_shape, 1, 'shape')
        #nx.set_node_attributes(current_shape, shape_number, 'shape_number')

        s = subgraph_generator()
        relabeler = {ns: floor_counter + ns for ns in s.nodes}
        s = nx.relabel.relabel_nodes(s, relabeler)
        nx.set_node_attributes(s, 0, 'shape')

        # Join s and shape together:
        to_pivot = random.choice(list(shape.nodes))
        pivot = random.choice(list(s.nodes))

        shape_node_per_subgraph.append(pivot) # This node represents the shape in the graph

        convert = {to_pivot: pivot}

        mx_nodes = max(list(s.nodes))
        i = 1
        for n in current_shape.nodes:
            if not (n == to_pivot):
                convert[n] = mx_nodes + i
            i += 1

        current_shape = nx.relabel.relabel_nodes(current_shape, convert)

        original_s_nodes = list(s.nodes)
        s.add_nodes_from(current_shape.nodes(data=True))
        s.add_edges_from(current_shape.edges)
        #print(original_s_nodes)

        #add_nodes = [current_shape.nodes(data=True)[n] for n in current_shape.nodes if n != pivot]

        #print(list(set(list(current_shape.nodes)) - set([pivot])))
        # s.add_nodes_from(list(set(list(current_shape.nodes)) - set([pivot])))
        # # s.add_nodes_from(add_nodes)
        # for n in current_shape.nodes: # Update properties of shape nodes
        #     s.nodes[n]['shapes_in_khop'] = 1
        # s.add_edges_from(list(current_shape.edges))

        # nx.draw(s)
        # plt.show()

        # Find k-hop from pivot:
        in_house = khop_subgraph_nx(node_idx = pivot, num_hops = num_hops, G = s)
        s.remove_nodes_from(set(s.nodes) - set(in_house) - set(current_shape.nodes))
        nx.set_node_attributes(s, 1, 'shapes_in_khop')

        # to_remove = []
        # for o in original_s_nodes:
        #     not_house_in_khop = check_khop_bound(s, 
        #         node=o, 
        #         num_hops = num_hops, 
        #         attr_measure = 'shapes_in_khop', 
        #         bound=0, 
        #         min_or_max=1) # Detect if a node is within the distance of

        #     # subgraph_nodes = khop_subgraph_nx(node_idx=o, num_hops = num_hops, G = s)
        #     # measures = [s.nodes[n][attr_measure] for n in subgraph_nodes]

        #     # if max(measures) < 1:

        #     #if house_in_khop: # Maximum value should be 1 (either 1 or zero)
        #         # Increment attribute for this node
        #         #s.nodes[o]['shapes_in_khop'] = 1
        #     if not_house_in_khop:
        #         # Remove any nodes not within distance of shape:
        #         to_remove.append(o)

        #print(to_remove)
        #s.remove_nodes_from(to_remove)

        # Separately assign house_in_khop:
        # for i in s.nodes:
        #     s.nodes[i]['shapes_in_khop'] = 1

        # nx.draw(s)
        # plt.title('After')
        # plt.show() 
                
        subgraphs.append(s)
        #print(s.nodes)
        floor_counter = max(list(s.nodes)) + 1
        original_shapes.append(current_shape.copy())

        shape_number += 1

    # for n in range(len(subgraphs)):
    #     nx.draw(subgraphs[n])
    #     plt.show()
    G = nx.Graph()
    #nx.set_node_attributes(G, 0, 'shapes')

    # all_edges = []
    # all_nodes = []
    for i in range(len(subgraphs)):
        # all_edges += list(subgraphs[i].edges)
        # all_nodes += list(subgraphs[i].nodes())
        G.add_edges_from(subgraphs[i].edges)
        G.add_nodes_from(subgraphs[i].nodes(data=True))


    # G.add_nodes_from(all_nodes)
    # G.add_edges_from(all_edges)

    # Join subgraphs via inner-subgraph connections
    # Rule: make 2 connections between any two graphs
    for i in range(len(subgraphs)):
        for j in range(i + 1, len(subgraphs)):
            #if i == j: # Don't connect the same subgraph
            #    continue

            s = subgraphs[i]
            # Try to make num_hops connections between subgraphs i, j:
            for k in range(inter_sg_connections):

                # Screen whether to try to make a connection:
                if np.random.rand() > prob_connection:
                    continue


                # # Store nodes previously held in subgraph of each shape:
                prev_nodes_i = khop_subgraph_nx(node_idx=shape_node_per_subgraph[i], num_hops = num_hops, G = G)
                prev_nodes_j = khop_subgraph_nx(node_idx=shape_node_per_subgraph[j], num_hops = num_hops, G = G)

                # Get all possible edges between the two subgraphs:
                x, y = np.meshgrid(list(subgraphs[i].nodes), list(subgraphs[j].nodes))
                possible_edges = list(zip(x.flatten(), y.flatten()))

                # Find random pair of nodes, search ahead to ensure bounds are not violated
                rand_edge = None
                while len(possible_edges) > 0:
                    rand_edge = random.choice(possible_edges)
                    possible_edges.remove(rand_edge) # Remove b/c we're searching this edge possibility

                    tempG = G.copy()

                    # Make edge between the two:
                    tempG.add_edge(*rand_edge)

                    # Union of all nodes touching the house nodes - for each subgraph:
                    temp_nodes1 = set()
                    for t in original_shapes[i].nodes:
                        temp_nodes1 = temp_nodes1.union(set(khop_subgraph_nx(node_idx = t, num_hops = num_hops, G = tempG)))
                    tempG = increment_shape_counter(
                        G = tempG, 
                        nodes = list(temp_nodes1), 
                        attr_measure = 'shapes_in_khop',
                        exclude_nodes = prev_nodes_i)

                    temp_nodes2 = set()
                    for t in original_shapes[j].nodes:
                        temp_nodes2 = temp_nodes2.union(set(khop_subgraph_nx(node_idx = t, num_hops = num_hops, G = tempG)))
                    tempG = increment_shape_counter(
                        G = tempG, 
                        nodes = list(temp_nodes2), 
                        attr_measure = 'shapes_in_khop',
                        exclude_nodes = prev_nodes_j)


                    # tempG = increment_shape_counter(
                    #     G = tempG, 
                    #     nodes = list(temp_nodes), 
                    #     attr_measure = 'shapes_in_khop',
                    #     exclude_nodes = list(set(prev_nodes_i).union(set(prev_nodes_j))))

                    # Increment shape counters for proper nodes:
                    # tempG = increment_shape_counter(tempG, insert_at_node = shape_node_per_subgraph[i], 
                    #     attr_measure='shapes_in_khop', 
                    #     num_hops = num_hops,
                    #     exclude_nodes = prev_nodes_i)

                    # tempG = increment_shape_counter(tempG, insert_at_node = shape_node_per_subgraph[j], 
                    #     attr_measure='shapes_in_khop', 
                    #     num_hops = num_hops,
                    #     exclude_nodes = prev_nodes_j)

                    # Check for violation:
                    if check_graph_bound(tempG, 'shapes_in_khop', 2, 1):
                        break

                    rand_edge = None # Would break out with this value if on final iteration

                if rand_edge is not None: # If we found a valid edge
                    #print('Made change')
                    G = tempG.copy()

    # Ensure that G is connected
    G = G.subgraph(sorted(nx.connected_components(G), key = len, reverse = True)[0])

    # Renumber nodes to be constantly increasing integers starting from 0
    # Unfreeze graph:
    mapping = {n:i for i, n in enumerate(G.nodes)}
    G = nx.relabel_nodes(G, mapping = mapping, copy = True)

    G = G.to_undirected()

    # Check the construction
    for t in G.nodes:
        nodes = khop_subgraph_nx(node_idx = t, num_hops = num_hops, G = G)
        nodes_in_khop = list(set(nodes) - set([t])) 
        count = Counter([G.nodes[n]['shape'] for n in nodes_in_khop])[1]
        if count != G.nodes[t]['shapes_in_khop']:
            print('node {}: count = {}, supposed = {}'.format(t, count, G.nodes[t]['shapes_in_khop'] - 1))


    # print(G.nodes)
    # print(mapping)
    # print(G.number_of_nodes())

    return G


if __name__ == '__main__':
    G = build_bound_graph(num_subgraphs = 20, num_hops=2)

    y = [d['shapes_in_khop'] for _, d in G.nodes(data=True)]
    print(np.unique(y))
    import pandas as pd
    print(pd.Series(y).value_counts())
    print('APL', nx.average_shortest_path_length(G))

    # nx.draw(G, node_color = y)
    # #plt.colorbar()
    # plt.show()

    from torch_geometric.utils import from_networkx
    from sklearn.model_selection import train_test_split

    data = from_networkx(G)

    x = []
    for n in G.nodes:
        x.append([G.degree(n), nx.clustering(G, nodes = n)])

    data.x = torch.tensor(x, dtype=torch.float32)
    data.y = torch.tensor(y, dtype=torch.long) - 1
    train_mask, test_mask = train_test_split(torch.tensor(range(data.x.shape[0])), 
        test_size = 0.2, stratify = data.y)
    train_tensor, test_tensor = torch.zeros(data.y.shape[0], dtype=bool), torch.zeros(data.y.shape[0], dtype=bool)
    train_tensor[train_mask] = 1
    test_tensor[test_mask] = 1

    data.train_mask = train_tensor
    print(data.train_mask)
    data.test_mask = test_tensor
    print(data.test_mask)

    model = GCN_3layer(128, input_feat=2, classes=2)

    count_0 = (data.y == 0).nonzero(as_tuple=True)[0].shape[0]
    count_1 = (data.y == 1).nonzero(as_tuple=True)[0].shape[0]

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    all_f1s = []
    all_acs = []
    all_prec = []
    all_rec = []
    for epoch in range(1,400):
        loss = train(model, optimizer, criterion, data)
        #print('Loss', loss.item())
        f1, acc, prec, rec = test(model, data)
        all_f1s.append(f1)
        all_acs.append(acc)
        all_prec.append(prec)
        all_rec.append(rec)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test F1: {f1:.4f}, Test Acc: {acc:.4f}, P: {prec:.4f}, R: {rec:.4f}')

    print('Count 0:', count_0)
    print('Count 1:', count_1)
    print('Max', max(all_f1s))
    print('Epochs:', np.argmax(all_f1s))

    x = list(range(len(all_f1s)))
    plt.plot(x, all_f1s, label = 'F1')
    plt.plot(x, all_acs, label = 'Accuracy')
    plt.plot(x, all_prec, label = 'Precision')
    plt.plot(x, all_rec, label = 'Recall')
    #plt.title('Metrics on {} ({} layers), {} Features'.format(sys.argv[2], sys.argv[3], sys.argv[1]))
    plt.xlabel('Epochs')
    plt.ylabel('Metric')
    plt.legend()
    plt.show()
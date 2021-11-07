import random
import numpy as np
import networkx as nx
from functools import partial
from collections import Counter
from .shapes import house
from graphxai.utils import khop_subgraph_nx
import matplotlib.pyplot as plt

from graphxai.gnn_models.node_classification.testing import *

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
    G = G.copy()
    for n in nodes_to_search:
        khop = khop_subgraph_nx(node_idx = n, num_hops = num_hops, G = G)

        unique_shapes = torch.unique(torch.tensor([G.nodes[i]['shape_number'] for i in khop]))
        num_unique = unique_shapes.shape[0] - 1 if 0 in unique_shapes else unique_shapes.shape[0]

        if num_unique < lower_bound or num_unique > upper_bound:
            return None
        else:
            G.nodes[n][attr_measure] = num_unique
            G.nodes[n]['nearby_shapes'] = unique_shapes

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
        nx.set_node_attributes(current_shape, shape_number, 'shape_number')

        s = subgraph_generator()
        relabeler = {ns: floor_counter + ns for ns in s.nodes}
        s = nx.relabel.relabel_nodes(s, relabeler)
        nx.set_node_attributes(s, 0, 'shape')
        nx.set_node_attributes(s, 0, 'shape_number')

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

        # Find k-hop from pivot:
        in_house = khop_subgraph_nx(node_idx = pivot, num_hops = num_hops, G = s)
        s.remove_nodes_from(set(s.nodes) - set(in_house) - set(current_shape.nodes))

        # Ensure that pivot is assigned to proper shape:
        s.nodes[pivot]['shape_number'] = shape_number
                
        subgraphs.append(s.copy())
        floor_counter = max(list(s.nodes)) + 1
        original_shapes.append(current_shape.copy())

        shape_number += 1

    G = nx.Graph()
    
    for i in range(len(subgraphs)):
        G.add_edges_from(subgraphs[i].edges)
        G.add_nodes_from(subgraphs[i].nodes(data=True))

    G = G.to_undirected()

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

                x, y = np.meshgrid(list(subgraphs[i].nodes), list(subgraphs[j].nodes))
                possible_edges = list(zip(x.flatten(), y.flatten()))

                rand_edge = None
                break_flag = False
                while len(possible_edges) > 0:

                    rand_edge = random.choice(possible_edges)
                    possible_edges.remove(rand_edge) # Remove b/c we're searching this edge possibility


                    tempG = G.copy()

                    # Make edge between the two:
                    tempG.add_edge(*rand_edge)
                    tempG.add_edge(rand_edge[1], rand_edge[0])

                    khop_union = set()

                    # Constant number of t's for each (10)
                    for t in list(original_shapes[i].nodes) + list(original_shapes[j].nodes):
                        khop_union = khop_union.union(set(khop_subgraph_nx(node_idx = t, num_hops = num_hops, G = tempG)))

                    tempG = incr_on_unique_houses(
                        nodes_to_search = list(khop_union),   
                        G = tempG, 
                        num_hops = num_hops, 
                        attr_measure = 'shapes_in_khop', 
                        lower_bound = 1, 
                        upper_bound = 2)

                    if tempG is None:
                        rand_edge = None
                        continue
                    else:
                        break

                if rand_edge is not None: # If we found a valid edge
                    #print('Made change')
                    G = tempG.copy()

    # Ensure that G is connected
    G = G.subgraph(sorted(nx.connected_components(G), key = len, reverse = True)[0])

    # Check the construction
    # number_off = 0
    # show_count = 0
    # for t in G.nodes:
    #     nodes = khop_subgraph_nx(node_idx = t, num_hops = num_hops, G = G)
    #     nodes_in_khop = list(set(nodes) - set([t])) 
    #     #count = Counter([G.nodes[n]['shape'] for n in nodes_in_khop])[1]
    #     unique = np.unique([G.nodes[n]['shape_number'] for n in nodes_in_khop])
    #     n_unique = len(unique) - 1 if 0 in unique else len(unique)

    #     if n_unique != G.nodes[t]['shapes_in_khop']:
    #         print('node {}: count = {}, supposed = {}'.format(t, n_unique, G.nodes[t]['shapes_in_khop']))
    #     if G.nodes[t]['shapes_in_khop'] != n_unique:
    #         #print('node {}: count = {}, supposed = {}'.format(t, n_unique, G.nodes[t]['shapes_in_khop']))
    #         print('off', unique)
    #         number_off += 1

    #     G.nodes[t]['shapes_in_khop'] = n_unique

    # print('Number off:', number_off)

    # Renumber nodes to be constantly increasing integers starting from 0
    mapping = {n:i for i, n in enumerate(G.nodes)}
    G = nx.relabel_nodes(G, mapping = mapping, copy = True)

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
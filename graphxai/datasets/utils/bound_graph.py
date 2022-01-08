import random
import numpy as np
import networkx as nx
from functools import partial
from typing import Optional, Callable, Union
from .shapes import house
from graphxai.utils import khop_subgraph_nx
import matplotlib.pyplot as plt

from graphxai.gnn_models.node_classification.testing import *

def incr_on_unique_houses(nodes_to_search, G, num_hops, attr_measure, lower_bound, upper_bound):
    #G = G.copy()

    incr_tuples = {}

    for n in nodes_to_search:
        khop = khop_subgraph_nx(node_idx = n, num_hops = num_hops, G = G)

        #unique_shapes = torch.unique(torch.tensor([G.nodes[i]['shape_number'] for i in khop]))
        unique_shapes = torch.unique(torch.tensor([G.nodes[i]['shape'] for i in khop]))
        num_unique = unique_shapes.shape[0] - 1 if 0 in unique_shapes else unique_shapes.shape[0]

        if num_unique < lower_bound or num_unique > upper_bound:
            return None
        else:

            incr_tuples[n] = (num_unique, unique_shapes)

            # G.nodes[n][attr_measure] = num_unique
            # G.nodes[n]['nearby_shapes'] = unique_shapes

    for k, v in incr_tuples.items():
        G.nodes[k][attr_measure] = v[0]
        G.nodes[k]['nearby_shapes'] = v[1]

    return G

def build_bound_graph(
        shape: Optional[nx.Graph] = house, 
        num_subgraphs: Optional[int] = 5, 
        inter_sg_connections: Optional[int] = 1,
        prob_connection: Optional[float] = 1,
        num_hops: Optional[int] = 2,
        base_graph: Optional[str] = 'ba',
        seed = None,
        **kwargs,
        ) -> nx.Graph:
    '''
    Creates a synthetic graph with one or two motifs within a given neighborhood and
        then labeling nodes based on the number of motifs around them. 
    Can be thought of as building unique explanations for each node, with either one
        or two motifs being the explanation.
    Args:
        shape (nx.Graph, optional): Motif to be inserted.
        num_subgraphs (int, optional): Number of initial subgraphs to create. Roughly
            controls number of nodes in the graph.
        inter_sg_connections (int, optional): How many connections to be made between
            subgraphs. Higher value will create more inter-connected graph. 
        prob_connection (float, optional): Probability of making connection between 
            subgraphs. Can introduce sparsity and stochasticity to graph generation.
        num_hops (int, optional): Number of hops to consider for labeling a node.
        base_graph (str, optional): Base graph algorithm used to generate each subgraph.
            Options are `'ba'` (Barabasi-Albert) (:default: :obj:`'ba'`)
    '''

    np.random.seed(seed)
    random.seed(seed)

    # Create graph:
    if base_graph == 'ba':
        if 'n_ba' in kwargs:
            subgraph_generator = partial(nx.barabasi_albert_graph, n=kwargs['n_ba'], m=1)
        else:
            subgraph_generator = partial(nx.barabasi_albert_graph, n=5 * num_hops, m=1)

    subgraphs = []
    shape_node_per_subgraph = []
    original_shapes = []
    floor_counter = 0
    shape_number = 1
    for i in range(num_subgraphs):
        current_shape = shape.copy()
        #nx.set_node_attributes(current_shape, 1, 'shape')
        #nx.set_node_attributes(current_shape, shape_number, 'shape_number')
        nx.set_node_attributes(current_shape, shape_number, 'shape')

        s = subgraph_generator()
        relabeler = {ns: floor_counter + ns for ns in s.nodes}
        s = nx.relabel.relabel_nodes(s, relabeler)
        nx.set_node_attributes(s, 0, 'shape')
        #nx.set_node_attributes(s, 0, 'shape_number')

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
        
        s.add_nodes_from(current_shape.nodes(data=True))
        s.add_edges_from(current_shape.edges)

        # Find k-hop from pivot:
        in_house = khop_subgraph_nx(node_idx = pivot, num_hops = num_hops, G = s)
        s.remove_nodes_from(set(s.nodes) - set(in_house) - set(current_shape.nodes))
        nx.set_node_attributes(s, 1, 'shapes_in_khop')

        # Ensure that pivot is assigned to proper shape:
        #s.nodes[pivot]['shape_number'] = shape_number
        s.nodes[pivot]['shape'] = shape_number


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
    for i in range(len(subgraphs)):
        for j in range(i + 1, len(subgraphs)):
            #if i == j: # Don't connect the same subgraph
            #    continue

            s = subgraphs[i]
            # Try to make connections between subgraphs i, j:
            for k in range(inter_sg_connections):

                # Screen whether to try to make a connection:
                if np.random.rand() > prob_connection:
                    continue

                x, y = np.meshgrid(list(subgraphs[i].nodes), list(subgraphs[j].nodes))
                possible_edges = list(zip(x.flatten(), y.flatten()))

                rand_edge = None

                tempG = G.copy()

                while len(possible_edges) > 0:

                    rand_edge = random.choice(possible_edges)
                    possible_edges.remove(rand_edge) # Remove b/c we're searching this edge possibility

                    # Make edge between the two:
                    tempG.add_edge(rand_edge[0], rand_edge[1])
                    tempG.add_edge(rand_edge[1], rand_edge[0])
                    #print('rand_edge 1', rand_edge)

                    khop_union = set()

                    # Constant number of t's for each (10)
                    for t in list(original_shapes[i].nodes) + list(original_shapes[j].nodes):
                        khop_union = khop_union.union(set(khop_subgraph_nx(node_idx = t, num_hops = num_hops, G = tempG)))

                    incr_ret = incr_on_unique_houses(
                        nodes_to_search = list(khop_union),   
                        G = tempG, 
                        num_hops = num_hops, 
                        attr_measure = 'shapes_in_khop', 
                        lower_bound = 1, 
                        upper_bound = 2)

                    if incr_ret is None:
                        #print('rand_edge 2', rand_edge)
                        tempG.remove_edge(rand_edge[0], rand_edge[1])
                        #tempG.remove_edge(rand_edge[1], rand_edge[0])

                        rand_edge = None
                        continue
                    else:
                        tempG = incr_ret
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
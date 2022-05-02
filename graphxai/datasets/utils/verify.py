import networkx as nx

def verify_motifs(G: nx.Graph, motif_subgraph: nx.Graph):
    '''
    Verifies that all motifs within a graph are "good" motifs
        i.e. they were planted by the building algorithm

    Args:
        G (nx.Graph): Networkx graph on which to search.
        motif_subgraph (nx.Graph): Motif to search for (query graph).

    Returns:
        :rtype: :obj:`bool`
        False if there exists at least one "bad" shape
        True if all motifs/shapes in the graph were planted
    '''

    matcher = nx.algorithms.isomorphism.ISMAGS(graph = G, subgraph = motif_subgraph)

    for iso in matcher.find_isomorphisms():
        nodes_found = iso.keys()
        shapes = [G.nodes[n]['shape'] for n in nodes_found]

        if (sum([int(shapes[i] != shapes[i-1]) for i in range(1, len(shapes))]) > 0) \
            or (sum(shapes) == 0):
            # Found a bad one
            return False

    return True

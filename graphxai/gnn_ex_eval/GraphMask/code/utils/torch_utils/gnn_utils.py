import torch


def add_backwards_edges(edges,
                        edge_types,
                        edge_embeddings=None,
                        n_relations=None,
                        separate_relation_types_for_inverse=None):
    backward_edges = torch.stack([edges[1], edges[0]], 0)
    direction_cutoff = int(edges.shape[1])

    edges = torch.cat([edges, backward_edges], -1)

    if edge_embeddings is not None:
        edge_embeddings = torch.cat([edge_embeddings, edge_embeddings], 0)

    if edge_types is not None:
        backwards_edge_types = edge_types.clone()
        if n_relations is not None and separate_relation_types_for_inverse:
            backwards_edge_types += n_relations

        edge_types = torch.cat([edge_types, backwards_edge_types], 0)

    return edges, edge_embeddings, edge_types, direction_cutoff

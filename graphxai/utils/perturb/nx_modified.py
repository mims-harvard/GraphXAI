"""
This code is modified from networkx.
"""
import torch
import random
import numpy as np
import networkx as nx

device = "cuda" if torch.cuda.is_available() else "cpu"

def swap(G, subset: list = None, nswap: int = 1,
         max_tries: int = 100, seed: int = None):
    """Swap two edges in the subgraph while keeping the node degrees fixed.

    A double-edge swap removes two randomly chosen edges u-v and x-y
    and creates the new edges u-x and v-y::

     u--v            u  v
            becomes  |  |
     x--y            x  y

    If either the edge u-x or v-y already exist no swap is performed
    and another attempt is made to find a suitable edge pair.

    Parameters
    ----------
    G : graph
       An undirected graph

    subset : list of integers
       Nodes in the subgraph

    nswap : integer (optional, default=1)
       Number of double-edge swaps to perform

    max_tries : integer (optional)
       Maximum number of attempts to swap edges

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    G : graph
       The graph after double edge swaps.

    Notes
    -----
    Does not enforce any connectivity constraints.

    The graph G is modified in place.
    """

    def cumulative_distribution(distribution, subset: list = None):
        """
        Returns normalized cumulative distribution from discrete distribution,
        restricted to the subset.
        """
        # Original implementation
        # cdf = [0.0]
        # if subset is not None:
        #     # Restrict the distribution to subset
        #     distribution = [d if i in subset else 0 for (i, d) in enumerate(distribution)]
        # psum = float(sum(distribution))
        # for i in range(0, len(distribution)):
        #     cdf.append(cdf[i] + distribution[i] / psum)

        # Tensorized implementation
        #cdf = [0.0]  # torch.Tensor([0.0])
        #dist_tensor = torch.from_numpy(np.asarray(distribution)).to(device)
        dist_tensor = torch.as_tensor(distribution).to(device)
        #temp_dist = torch.zeros(len(distribution)).long().to(device)
        temp_dist = torch.zeros(len(distribution)).long().to(device)
        if subset is not None:
            temp_dist[subset] = dist_tensor[subset]

        psum = float(sum(temp_dist))
        temp_dist = temp_dist/psum
        # for i in range(0, len(temp_dist)):
        #     # import ipdb; ipdb.set_trace()
        #     cdf.append(cdf[i] + temp_dist[i].item())
            # cdf = torch.cat((cdf, torch.Tensor([cdf[i]+temp_dist[i]])), dim=-1)
        #return cdf  # .numpy()
        return torch.cumsum(temp_dist, dim = 0).tolist()

    # Initialize seed for random
    random.seed(seed)

    if G.is_directed():
        raise nx.NetworkXError("double_edge_swap() not defined for directed graphs.")
    if nswap > max_tries:
        raise nx.NetworkXError("Number of swaps > number of tries allowed.")
    if len(G) < 4:
        raise nx.NetworkXError("Graph has less than four nodes.")
    # Instead of choosing uniformly at random from a generated edge list,
    # this algorithm chooses nonuniformly from the set of nodes with
    # probability weighted by degree.
    n = 0
    swapcount = 0
    keys, degrees = zip(*G.degree())  # keys, degree
    # import time; st_time = time.time()
    cdf = cumulative_distribution(degrees, subset)  # cdf of degree
    # print(f'{time.time()-st_time}')
    discrete_sequence = nx.utils.discrete_sequence
    while swapcount < nswap:
        #        if random.random() < 0.5: continue # trick to avoid periodicities?
        # pick two random edges without creating edge list
        # choose source node indices from discrete distribution
        (ui, xi) = discrete_sequence(2, cdistribution=cdf, seed=seed)
        if ui == xi:
            continue  # same source, skip
        u = keys[ui]  # convert index to label
        x = keys[xi]
        # choose target uniformly from neighbors
        v = random.choice(list(G[u]))
        y = random.choice(list(G[x]))
        if v == y:
            continue  # same target, skip
        if (x not in G[u]) and (y not in G[v]):  # don't create parallel edges
            G.add_edge(u, x)
            G.add_edge(v, y)
            G.remove_edge(u, v)
            G.remove_edge(x, y)
            swapcount += 1
        if n >= max_tries:
            e = (
                f"Maximum number of swap attempts ({n}) exceeded "
                f"before desired swaps achieved ({nswap})."
            )
            raise nx.NetworkXAlgorithmError(e)
        n += 1
    return G

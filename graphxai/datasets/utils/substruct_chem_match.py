import torch
import networkx as nx

from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, subgraph
from torch_geometric.utils.convert import to_networkx

# MUTAG notation: [C, N, O, ...]

# MUTAG notation:
# 0	C
# 1	O
# 2	Cl
# 3	H
# 4	N
# 5	F
# 6	Br
# 7	S
# 8	P
# 9	I
# 10	Na
# 11	K
# 12	Li
# 13	Ca

# C_vec = torch.zeros(7); C_vec[0] = 1
# N_vec = torch.zeros(7); N_vec[1] = 1
# O_vec = torch.zeros(7); O_vec[2] = 1 


C_vec = torch.zeros(14); C_vec[0] = 1
O_vec = torch.zeros(14); O_vec[1] = 1 
Cl_vec = torch.zeros(14); Cl_vec[2] = 1
H_vec = torch.zeros(14); H_vec[3] = 1
N_vec = torch.zeros(14); N_vec[4] = 1
F_vec = torch.zeros(14); F_vec[5] = 1
Br_vec = torch.zeros(14); Br_vec[6] = 1
I_vec = torch.zeros(14); I_vec[9] = 1

def make_NO2():
    no2 = nx.Graph()

    nodes = [0, 1, 2]
    atom = ['N', 'O', 'O']
    vecs = [N_vec, O_vec, O_vec]

    node_data = [(n, {'atom': a, 'x': v.clone()}) for n, a, v in zip(nodes, atom, vecs)]

    no2.add_nodes_from(node_data)

    # Edges like: O -- N -- O
    edges = [(0, 1), (0, 2)]

    no2.add_edges_from(edges)

    return no2

def make_NH2():
    nh2 = nx.Graph()

    nodes = [0, 1, 2]
    atom = ['N', 'H', 'H']
    vecs = [N_vec, H_vec, H_vec]

    node_data = [(n, {'atom': a, 'x': v.clone()}) for n, a, v in zip(nodes, atom, vecs)]

    nh2.add_nodes_from(node_data)

    # Edges like: H -- N -- H
    edges = [(0, 1), (0, 2)]

    nh2.add_edges_from(edges)

    return nh2

MUTAG_NO2 = make_NO2()
MUTAG_NH2 = make_NH2()

def match_NH2(G: nx.Graph, node: int):
    '''Determines if a node in a Networkx graph is a match for NH2 group'''

    if G.degree[node] != 1: # If degree not 1, isn't NH2
        return None

    # See if their vectors are equal:
    node_vec = torch.as_tensor(G.nodes[node]['x'])
    if (torch.norm(node_vec - N_vec).item() == 0):
        return node
    else:
        return None
    #return (torch.norm(node_vec - N_vec).item() == 0)

    # if (torch.norm(node_vec - gt_vec).item() == 0):
    #     # Highlight the node
    #     pass

    # return None

def match_substruct(G: nx.Graph, substructure: nx.Graph = MUTAG_NO2):


    # Isomorphic match graph structure:
    matcher = nx.algorithms.isomorphism.ISMAGS(graph = G, subgraph = substructure)

    matches = []

    for iso in matcher.find_isomorphisms():
        #i += 1

        # iso is a dictionary (G nodes -> sub nodes)

        bad_iso_flag = False
        
        # Find matching to 0
        # Ensure the subgraph matches attributes (i.e. N's, O's, H's, etc.)
        for k, v in iso.items():
            if v != 0: continue
            
            # Make sure k is N:
            if G.degree[k] != 3:
                bad_iso_flag = True
                break
        
            # If this node is not N:
            node_vec = torch.as_tensor(G.nodes[k]['x'])
            if torch.norm(node_vec - N_vec).item() != 0:
                bad_iso_flag = True
                break

            # Get all oxygen nodes:
            O_nodes = [ki for ki, _ in iso.items() if ki != k]

            # Check if they're both Oxygens and if O_nodes only have degree 1
            for o in O_nodes:

                if G.degree[o] != 1:
                    bad_iso_flag = True
                    break

                O_vec_i = torch.as_tensor(G.nodes[o]['x'])

                if torch.norm(O_vec_i - O_vec).item() != 0:
                    bad_iso_flag = True
                    break
            
            # Break no matter what here
            break

        if bad_iso_flag:
            continue
        
        # Mark each of the nodes:
        matches.append(torch.as_tensor(list(iso.keys()), dtype = int))

    return matches

def match_substruct_mutagenicity(G: nx.Graph, substructure: nx.Graph = MUTAG_NO2, nh2_no2 = 0):
    '''
    Only works for NH2 and NO2

    Args:
        nh2_no2: 0 if NH2, 1 if NO2
    '''

    # Isomorphic match graph structure:
    matcher = nx.algorithms.isomorphism.ISMAGS(graph = G, subgraph = substructure)

    matches = []

    for iso in matcher.find_isomorphisms():
        #i += 1

        # iso is a dictionary (G nodes -> sub nodes)

        bad_iso_flag = False
        
        # Find matching to 0
        # Ensure the subgraph matches attributes (i.e. N's, O's, H's, etc.)
        for k, v in iso.items():
            if v != 0: continue # Ensures we're checking what maps to v=0
            
            # Make sure k is N:
            if G.degree[k] != 3:
                bad_iso_flag = True
                break
        
            # If this node is not N:
            node_vec = torch.as_tensor(G.nodes[k]['x'])
            if torch.norm(node_vec - N_vec).item() != 0:
                bad_iso_flag = True
                break

            # Get all oxygen nodes:
            O_nodes = [ki for ki, _ in iso.items() if ki != k]

            # Check if they're both Oxygens and if O_nodes only have degree 1
            for o in O_nodes:

                if G.degree[o] != 1:
                    bad_iso_flag = True
                    break

                O_vec_i = torch.as_tensor(G.nodes[o]['x'])

                if nh2_no2 == 0:
                    if torch.norm(O_vec_i - H_vec).item() != 0:
                        bad_iso_flag = True
                        break
                elif nh2_no2 == 1:
                    if torch.norm(O_vec_i - O_vec).item() != 0:
                        bad_iso_flag = True
                        break
            
            # Break no matter what here
            break

        if bad_iso_flag:
            continue
        
        # Mark each of the nodes:
        matches.append(torch.as_tensor(list(iso.keys()), dtype = int))

    return matches

def match_aliphatic_halide(G: nx.Graph):
    '''
    Matches -Cl,Br,I group
    '''
    matches = []

    all_cycles = nx.cycle_basis(G)
    
    for n, data in G.nodes(data=True):
        
        if G.degree[n] > 1:
            continue

        other_end =[i_n for i_n in list(G.edges(n))[0] if i_n != n][0]

        for c in all_cycles:
            if (other_end in c) and len(c) > 4: # Roughly determines if connected atom is in aromatic ring?
                break
        else:
            continue

        # Test if the x matches Cl, Br, or I
        X = torch.as_tensor(data['x']).int()
        #print(X)
        if torch.norm(X - Cl_vec).item() == 0 or \
            torch.norm(X - Br_vec).item() == 0 or \
            torch.norm(X - I_vec).item() == 0 or \
            torch.norm(X - F_vec).item() == 0:
            matches.append(n)
    
    return matches

def match_nitroso(G: nx.Graph):
    '''
    
    Nitroso: R--N==O
    
    '''
    matches = []

    # Try all edges, any isomorphic search will just get any edge
    # ASSUMPTION: G doesn't have any duplicate edges (shouldn't be true unless MultiDiGraph) 
    for e1, e2 in G.edges():
        e1X = torch.as_tensor(G.nodes[e1]['x'])
        e2X = torch.as_tensor(G.nodes[e2]['x'])

        if G.degree[e1] == 2 and G.degree[e2] == 1:
            if (torch.norm(e1X - N_vec).item() == 0) and \
                (torch.norm(e2X - O_vec).item() == 0):
                matches.append(torch.as_tensor([e1, e2], dtype = int))

        elif G.degree[e2] == 2 and G.degree[e1] == 1:
            if (torch.norm(e2X - N_vec).item() == 0) and \
                (torch.norm(e1X - O_vec).item() == 0):
                matches.append(torch.as_tensor([e1, e2], dtype = int))

        # Else, we know we don't have it

    return matches  

def match_azo_type(G: nx.Graph):


    matches = []

    for e1, e2 in G.edges():

        e1X = torch.as_tensor(G.nodes[e1]['x'])
        e2X = torch.as_tensor(G.nodes[e2]['x'])

        if (G.degree[e1] == G.degree[e2]) and (G.degree[e2] == 2) and \
                torch.norm(e1X - N_vec).item() == 0 and \
                torch.norm(e2X - N_vec).item() == 0:
            matches.append(torch.as_tensor([e1, e2], dtype = int))

    return matches
            
def match_polycyclic(G: nx.Graph):

    '''
    Find any group of 3 or more conjoined rings
    '''

    cycles = list(nx.cycle_basis(G))

    if len(cycles) < 3:
        return []

    # Get all nodes, make subgraph:
    all_nodes = []
    for i in range(len(cycles)):
        for j in range(len(cycles[i])):
            all_nodes.append(cycles[i][j])

    subG = G.subgraph(all_nodes)
    conn_comp = sorted(nx.connected_components(subG), key=len, reverse=True)

    matches = []
    
    # Go over connected components, from largest to smallest:
    for i in range(len(conn_comp)):
        this_subG = G.subgraph(conn_comp[i])

        cycles = list(nx.cycle_basis(this_subG))

        # Filter for cycles that are less than 5 atoms:
        to_del = [i for i in range(len(cycles)) if len(cycles[i]) < 5]
        #copy_cycles = 
        for d in to_del:
            del cycles[d]


        if len(cycles) >= 3:
            matches.append(torch.as_tensor(list(this_subG.nodes), dtype = int))

    return matches



def match_NO2_old(data):
    '''
    Identifies edges and nodes in a graph that correspond to NO2 groups

    Args:
        data (torch_geometric.data.Data): One graph on which to match for
            NO2 groups

    Returns:
        List of (subgraph nodes (Tensor), edge mask (Tensor))
    '''

    isN = []
    for i in range(data.x.shape[0]):
        # We know that index 1 in x vec is N (one-hot)
        if data.x[i,1].item() == 1:
            isN.append(i)

    # Get k-hop subgraph around all N's
    # Check if that subgraph contains two O's with only two 
    #   edges in the edge_index (since its undirected)

    ground_truths = []

    for N in isN:
        subset, _, _, _ = k_hop_subgraph()

        # 1. make sure there's two O's:
        Os = []
        for sub_node in subset.tolist():
            # We know that index 2 in x is O (one-hot)
            if data.x[sub_node,2].item() == 1:
                Os.append(sub_node)

        if len(Os) != 2:
            # Needs to have two O's
            break

        # Examine the Os:
        Os_pass = True
        for O in Os:
            # Count the number of occurences in each row of 
            #   the edge index:
            num_0 = torch.sum(data.edge_index[0,:] == O).item()
            num_1 = torch.sum(data.edge_index[1,:] == O).item()

            if not (num_0 == num_1 == 1):
                Os_pass = False # Set flag

        if Os_pass: # Know that we have a hit
            subgraph_nodes = torch.tensor([N] + Os, dtype = int)

            _, _, edge_mask = subgraph(
                subgraph_nodes, 
                edge_index = data.edge_index,
                return_edge_mask = True,
                )

            ground_truths.append((subgraph_nodes, edge_mask))

    return ground_truths
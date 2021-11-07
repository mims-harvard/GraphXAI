from networkx.classes import graph
import torch
import networkx as nx
import matplotlib.pyplot as plt

from torch_geometric.utils import from_networkx, k_hop_subgraph, subgraph
from torch_geometric.data import Data

import graphxai.utils as gxai_utils
from graphxai.utils.nx_conversion import match_torch_to_nx_edges, remove_duplicate_edges

from typing import Optional

class EnclosingSubgraph:
    '''
    Args: 
        nodes (torch.LongTensor): Nodes in subgraph.
        edge_index (torch.LongTensor): Edge index for subgraph 
        inv (torch.LongTensor): Inversion of nodes in subgraph (see
            torch_geometric.utils.k_hop_subgraph method.)
        edge_mask (torch.BoolTensor): Mask of edges in entire graph.
        directed (bool, optional): If True, subgraph is directed. 
            (:default: :obj:`False`)
    '''
    def __init__(self, 
            nodes: torch.LongTensor, 
            edge_index: torch.LongTensor, 
            inv: torch.LongTensor, 
            edge_mask: torch.BoolTensor, 
            directed: Optional[bool] = False
        ):

        self.nodes = nodes
        self.edge_index = edge_index
        self.inv = inv
        self.edge_mask = edge_mask
        self.directed = directed

    def draw(self, show = False):
        G = gxai_utils.to_networkx_conv(Data(edge_index=self.edge_index), to_undirected=True)
        nx.draw(G)
        if show:
            plt.show()

class WholeGraph:

    def __init__(self, x = None, edge_index = None, y = None, directed = False):
        self.setup(x, edge_index, y, directed)

    def setup(self, x, edge_index, y = None, directed = False):
        '''
        Post-instantiation function to set member variables.
        '''
        # post-init function to set up class variables
        self.x = x
        self.edge_index = edge_index
        self.y = y
        self.directed = directed

    def get_Data(self):
        return Data(x=self.x, edge_index=self.edge_index, y=self.y)

    def to_networkx_conv(self, 
        to_undirected=False, 
        remove_self_loops: Optional[bool]=False,
        get_map: Optional[bool] = False):

        if to_undirected:
            G = nx.Graph()
        else:
            G = nx.DiGraph()

        node_list = sorted(torch.unique(self.edge_index).tolist())
        map_norm =  {node_list[i]:i for i in range(len(node_list))}

        G.add_nodes_from([map_norm[n] for n in node_list])

        # Assign values to each node:
        # Skipping for now

        for i, (u, v) in enumerate(self.edge_index.t().tolist()):
            u = map_norm[u]
            v = map_norm[v]

            if to_undirected and v > u:
                continue

            if remove_self_loops and u == v:
                continue

            G.add_edge(u, v)

            # No edge_attr additions added now
            # for key in edge_attrs if edge_attrs is not None else []:
            #     G[u][v][key] = values[key][i]

        if get_map:
            return G, map_norm

        return G


class Explanation:
    '''
    Members:
        feature_imp (torch.Tensor): Feature importance scores
            - Size: (x1,) with x1 = number of features
        node_imp (torch.Tensor): Node importance scores
            - Size: (n,) with n = number of nodes in subgraph or graph
        edge_imp (torch.Tensor): Edge importance scores
            - Size: (e,) with e = number of edges in subgraph or graph
        node_idx (int): Index for node explained by this instance
        node_reference (tensor of ints): Tensor matching length of `node_reference` 
            which maps each index onto original node in the graph
        edge_reference (tensor of ints): Tensor maching lenght of `edge_reference`
            which maps each index onto original edge in the graph's edge
            index
        graph (torch_geometric.data.Data): Original graph on which explanation
            was computed
            - Optional member, can be left None if graph is too large
    Optional members:
        enc_subgraph (Subgraph): k-hop subgraph around 
            - Corresponds to nodes and edges comprising computational graph around node
    '''
    def __init__(self,
        feature_imp: Optional[torch.tensor] = None,
        node_imp: Optional[torch.tensor] = None,
        edge_imp: Optional[torch.tensor] = None,
        node_idx: Optional[torch.tensor] = None,
        node_reference: Optional[torch.tensor] = None,
        edge_reference: Optional[torch.tensor] = None,
        graph = None):
        # Establish basic properties
        self.feature_imp = feature_imp
        self.node_imp = node_imp
        self.edge_imp = edge_imp

        # Only established if passed explicitly in init, not overwritten by enclosing subgraph 
        #   unless explicitly specified
        self.node_reference = node_reference
        self.edge_reference = edge_reference

        self.node_idx = node_idx # Set this for node-level prediction explanations
        self.graph = graph

    def set_enclosing_subgraph(self, subgraph, set_references = False):
        '''
        Args:
            k_hop_tuple (tuple, EnclosingSubgraph, or nx.Graph): Return value from torch_geometric.utils.k_hop_subgraph
        '''
        if isinstance(subgraph, EnclosingSubgraph):
            self.enc_subgraph = subgraph
        elif isinstance(subgraph, nx.Graph):
            # Convert from nx.Graph
            data = from_networkx(subgraph)
            nodes = torch.unique(data.edge_index)
            # TODO: Support inv and edge_mask through networkx
            self.enc_subgraph = EnclosingSubgraph(
                nodes = nodes,
                edge_index = data.edge_index,
                inv = None,
                edge_mask = None
            )
        else: # Assumed to be a tuple:
            self.enc_subgraph = EnclosingSubgraph(*subgraph)

        #if set_references or 
        if self.node_reference is None:
            self.node_reference = gxai_utils.make_node_ref(self.enc_subgraph.nodes)
            #self.edge_reference = self.enc_subgraph.edge_index

    def apply_subgraph_mask(self, 
        mask_node: Optional[bool] = False, 
        mask_edge: Optional[bool] = False):
        '''
        Performs automatic masking on the node and edge importance members

        Args:
            mask_node (bool, optional): If True, performs masking on node_imp based on enclosing subgraph nodes.
                Assumes that node_imp is set for entire graph and then applies mask.
            mask_edge (bool, optional): If True, masks edges in edge_imp based on enclosing subgraph edge mask.

        Example workflow:
        >>> exp = Explanation()
        >>> exp.node_imp = node_importance_tensor
        >>> exp.edge_imp = edge_importance_tensor
        >>> exp.set_enclosing_subgraph(k_hop_subgraph(node_idx, k, edge_index))
        >>> exp.apply_subgraph_mask(True, True) # Masks both node and edge importance
        '''
        if mask_edge:
            mask_inds = self.enc_subgraph.edge_mask.nonzero(as_tuple=True)[0]
            self.edge_imp = self.edge_imp[mask_inds] # Perform masking on current member
        if mask_node:
            self.node_imp = self.node_imp[self.enc_subgraph.nodes]

    def set_whole_graph(self, x, edge_index):
        self.graph = WholeGraph(x, edge_index)

    def graph_to_networkx(self, 
        to_undirected=False, 
        remove_self_loops: Optional[bool]=False,
        get_map: Optional[bool] = False):
        '''
        Convert graph to Networkx Graph

        Args:
            to_undirected (bool, optional): If True, graph is undirected. (:default: :obj:`False`)
            remove_self_loops (bool, optional): If True, removes all self-loops in graph.
                (:default: :obj:`False`)
            get_map (bool, optional): If True, returns a map of nodes in graph 
                to nodes in the Networkx graph. (:default: :obj:`False`)

        :rtype: :class:`Networkx.Graph` or :class:`Networkx.DiGraph`
            If `get_map == True`, returns tuple: (:class:`Networkx.Graph`, :class:`dict`)
        '''

        if to_undirected:
            G = nx.Graph()
        else:
            G = nx.DiGraph()

        node_list = sorted(torch.unique(self.graph.edge_index).tolist())
        map_norm =  {node_list[i]:i for i in range(len(node_list))}

        G.add_nodes_from([map_norm[n] for n in node_list])

        # Assign values to each node:
        # Skipping for now

        for i, (u, v) in enumerate(self.graph.edge_index.t().tolist()):
            u = map_norm[u]
            v = map_norm[v]

            if to_undirected and v > u:
                continue

            if remove_self_loops and u == v:
                continue

            G.add_edge(u, v)

            # No edge_attr additions added now
            if self.edge_imp is not None:
                G[u][v]['edge_imp'] = self.edge_imp[i].item()
            # for key in edge_attrs if edge_attrs is not None else []:
            #     G[u][v][key] = values[key][i]

        if self.node_imp is not None:
            for i, feat_dict in G.nodes(data=True):
                # self.node_imp[i] should be a scalar value
                feat_dict.update({'node_imp': self.node_imp[map_norm[i]].item()})

        if get_map:
            return G, map_norm

        return G

    def enc_subgraph_to_networkx(self, 
        to_undirected=False, 
        remove_self_loops: Optional[bool]=False,
        get_map: Optional[bool] = False):
        '''
        Convert enclosing subgraph to Networkx Graph

        Args:
            to_undirected (bool, optional): If True, graph is undirected. (:default: :obj:`False`)
            remove_self_loops (bool, optional): If True, removes all self-loops in graph.
                (:default: :obj:`False`)
            get_map (bool, optional): If True, returns a map of nodes in enclosing subgraph 
                to nodes in the Networkx graph. (:default: :obj:`False`)

        :rtype: :class:`Networkx.Graph` or :class:`Networkx.DiGraph`
            If `get_map == True`, returns tuple: (:class:`Networkx.Graph`, :class:`dict`)
        '''

        if to_undirected:
            G = nx.Graph()
        else:
            G = nx.DiGraph()

        node_list = sorted(torch.unique(self.enc_subgraph.edge_index).tolist())
        map_norm =  {node_list[i]:i for i in range(len(node_list))}
        rev_map = {v:k for k,v in map_norm.items()}

        G.add_nodes_from([map_norm[n] for n in node_list])

        # Assign values to each node:
        # Skipping for now

        for i, (u, v) in enumerate(self.enc_subgraph.edge_index.t().tolist()):
            u = map_norm[u]
            v = map_norm[v]

            if to_undirected and v > u:
                continue

            if remove_self_loops and u == v:
                continue

            G.add_edge(u, v)

            if self.edge_imp is not None:
                G.edges[u, v]['edge_imp'] = self.edge_imp[i].item()

            # No edge_attr additions added now
            # for key in edge_attrs if edge_attrs is not None else []:
            #     G[u][v][key] = values[key][i]

        if self.node_imp is not None:
            for i, feat_dict in G.nodes(data=True):
                # self.node_imp[i] should be a scalar value
                feat_dict.update({'node_imp': self.node_imp[i].item()})

        if get_map:
            return G, map_norm

        return G

    def context_draw(self, 
            num_hops,
            additional_hops = 1, 
            heat_by_prescence = False, 
            heat_by_exp = True, 
            ax = None,
            show=False
        ):
        '''
        Shows the explanation in context of a few more hops out than its k-hop neighborhood
        Args:
        '''

        data_G = self.graph.get_Data()
        wholeG = gxai_utils.to_networkx_conv(data_G, to_undirected=True)
        kadd_hop_neighborhood = gxai_utils.khop_subgraph_nx(
                G = wholeG, 
                num_hops= num_hops + additional_hops, 
                node_idx=self.node_idx
            )

        subG = wholeG.subgraph(kadd_hop_neighborhood)

        # Identify highlighting nodes:
        exp_nodes = self.enc_subgraph.nodes

        draw_args = dict()

        if heat_by_prescence:
            if self.node_imp is not None:
                node_c = [int(i in exp_nodes) for i in subG.nodes]
                draw_args['node_color'] = node_c

        if heat_by_exp:
            if self.node_imp is not None:
                node_c = []
                for i in subG.nodes:
                    if i in self.enc_subgraph.nodes:
                        node_c.append(self.node_imp[self.node_reference[i]])
                    else:
                        node_c.append(0)

                draw_args['node_color'] = node_c

            if self.edge_imp is not None:
                whole_edge_index, _ = subgraph(kadd_hop_neighborhood, edge_index = data_G.edge_index)

                # Need to match edge indices across edge_index and edges in graph
                tuple_edge_index = [(whole_edge_index[0,i].item(), whole_edge_index[1,i].item()) \
                    for i in range(whole_edge_index.shape[1])]

                print(len(tuple_edge_index), self.edge_imp.shape)

                trimmed_enc_subg_edge_index, emask = remove_duplicate_edges(self.enc_subgraph.edge_index)
                positive_edge_indices = self.edge_imp[emask].nonzero(as_tuple=True)[0]
                positive_edges = [(trimmed_enc_subg_edge_index[0,e].item(), trimmed_enc_subg_edge_index[1,e].item()) \
                    for e in positive_edge_indices]

                # Tuples in list should be hashable
                edge_list = list(subG.edges)

                # Get dictionary with mapping from edge index to networkx graph
                edge_matcher = match_torch_to_nx_edges(subG, remove_duplicate_edges(whole_edge_index)[0])

                edge_heat = torch.zeros(len(edge_list))

                for e in positive_edges:
                    # Must find index, which is not very efficient
                    edge_heat[edge_matcher[e]] = 1

                draw_args['edge_color'] = edge_heat.tolist()
                #coolwarm
                draw_args['edge_cmap'] = plt.cm.coolwarm

            # Heat edge explanations if given

        # Seed the position to stay consistent:
        pos = nx.spring_layout(subG, seed = 1234)
        nx.draw(subG, pos, ax = ax, **draw_args)

        # Highlight the node index:
        nx.draw(subG.subgraph(self.node_idx), pos, node_color = 'red', 
                node_size = 400, ax = ax)

        if show:
            plt.show()
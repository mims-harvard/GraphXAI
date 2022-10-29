from networkx.classes import graph
import torch
import torch_geometric
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt

from torch_geometric.utils import from_networkx, k_hop_subgraph, subgraph
from torch_geometric.data import Data

import graphxai.utils as gxai_utils
from graphxai.utils.nx_conversion import match_torch_to_nx_edges, remove_duplicate_edges
from graphxai.utils.misc import top_k_mask, threshold_mask

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

    def set_enclosing_subgraph(self, subgraph):
        '''
        Args:
            subgraph (tuple, EnclosingSubgraph, or nx.Graph): Return value from torch_geometric.utils.k_hop_subgraph
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

        if self.node_reference is None:
            self.node_reference = gxai_utils.make_node_ref(self.enc_subgraph.nodes)

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

    def set_whole_graph(self, data: Data):
        '''
        Args:
            data (torch_geometric.data.Data): Data object representing the graph to store.
        
        :rtype: :obj:`None`
        '''
        self.graph = data

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

    def top_k_node_imp(self, top_k: int, inplace = False):
        '''
        Top-k masking of the node importance for this Explanation.

        Args:
            top_k (int): How many highest scores to include in the mask.
            inplace (bool, optional): If True, masks the node_imp member 
                of the class.

        :rtype: :obj:`torch.Tensor`
        '''

        if inplace:
            self.node_imp = top_k_mask(self.node_imp, top_k)
        else:
            return top_k_mask(self.node_imp, top_k)

    def top_k_edge_imp(self, top_k: int, inplace = False):
        '''
        Top-k masking of the edge importance for this Explanation.

        Args:
            top_k (int): How many highest scores to include in the mask.
            inplace (bool, optional): If True, masks the node_imp member 
                of the class.

        :rtype: :obj:`torch.Tensor`
        '''
        if inplace:
            self.edge_imp = top_k_mask(self.edge_imp, top_k)
        else:
            return top_k_mask(self.edge_imp, top_k)

    def top_k_feature_imp(self, top_k: int, inplace = False):
        '''
        Top-k masking of the feature importance for this Explanation.

        Args:
            top_k (int): How many highest scores to include in the mask.
            inplace (bool, optional): If True, masks the node_imp member 
                of the class.

        :rtype: :obj:`torch.Tensor`
        '''
        if inplace:
            self.feature_imp = top_k_mask(self.feature_imp, top_k)
        else:
            return top_k_mask(self.feature_imp, top_k)

    def thresh_node_imp(self, threshold: float, inplace = False):
        '''
        Threshold mask the node importance

        Args:
            threshold (float): Select all values greater than this value.
            inplace (bool, optional): If True, masks the node_imp member 
                of the class.

        :rtype: :obj:`torch.Tensor`
        '''
        if inplace:
            self.node_imp = threshold_mask(self.node_imp, threshold)
        else:
            return threshold_mask(self.node_imp, threshold)

    def thresh_edge_imp(self, threshold: float, inplace = False):
        '''
        Threshold mask the edge importance

        Args:
            threshold (float): Select all values greater than this value.
            inplace (bool, optional): If True, masks the node_imp member 
                of the class.

        :rtype: :obj:`torch.Tensor`
        '''
        if inplace:
            self.edge_imp = threshold_mask(self.edge_imp, threshold)
        else:
            return threshold_mask(self.edge_imp, threshold)

    def thresh_feature_imp(self, threshold: float, inplace = False):
        '''
        Threshold mask the feature importance

        Args:
            threshold (float): Select all values greater than this value.
            inplace (bool, optional): If True, masks the node_imp member 
                of the class.

        :rtype: :obj:`torch.Tensor`
        '''
        if inplace:
            self.feature_imp = threshold_mask(self.feature_imp, threshold)
        else:
            return threshold_mask(self.feature_imp, threshold)

    def visualize_node(self, 
            num_hops: int,
            graph_data: Data = None,
            additional_hops: int = 1, 
            heat_by_prescence: bool = False, 
            heat_by_exp: bool = True, 
            node_agg_method: str = 'sum',
            ax: matplotlib.axes.Axes = None,
            show: bool = False,
            show_node_labels: bool = False,
            norm_imps = False,
        ):
        '''
        Shows the explanation in context of a few more hops out than its k-hop neighborhood. Used for
            visualizing the explanation for a node-level prediction task.
        
        ..note:: If neither `heat_by_prescence` or `heat_by_exp` are true, the method plots a simple
            visualization of the subgraph around the focal node.

        Args:
            num_hops (int): Number of hops in the enclosing subgraph.
            graph_data (torch_geometric.data.Data, optional): Data object containing graph. Don't provide
                if already stored in the dataset. Used so large graphs can be stored externally and used
                for visualization. (:default: :obj:`None`)
            additional_hops (int, optional): Additional number of hops to include for the visualization.
                If the size of the enclosing subgraph for a node `v` with respect to some model `f` 
                is `n`, then we would show the `n + additional_hops`-hop neighborhood around `v`.
                (:default: :obj:`1`)
            heat_by_prescence (bool, optional): If True, only highlights nodes in the enclosing subgraph.
                Useful for debugging or non-explanation visualization. (:default: :obj:`False`)
            heat_by_exp (bool, optional): If True, highlights nodes and edges by explanation values. 
                (:default: :obj:`True`)
            node_agg_method (str, optional): Aggregation method to use for showing multi-dimensional
                node importance scores (i.e. across features, such as GuidedBP or Vanilla Gradient).
                Options: :obj:`'sum'` and :obj:`'max'`. (:default: :obj:`'sum'`)
            ax (matplotlib.axes.Axes, optional): Axis on which to draw. If not provided, draws directly
                to plt. (:default: :obj:`None`)
            show (bool, optional): If True, shows the plot immediately after drawing. (:default: :obj:`False`)
            show_node_labels (bool, optional): If True, shows the node labels as integers overlaid on the 
                plot. (:default: :obj:`False`)
        '''

        assert self.node_idx is not None, "visualize_node only for node-level explanations, but node_idx is None" 

        #data_G = self.graph.get_Data()
        wholeG = gxai_utils.to_networkx_conv(graph_data, to_undirected=True)
        kadd_hop_neighborhood = gxai_utils.khop_subgraph_nx(
                G = wholeG, 
                num_hops= num_hops + additional_hops, 
                node_idx=self.node_idx
            )

        subG = wholeG.subgraph(kadd_hop_neighborhood)

        node_agg = torch.sum if node_agg_method == 'sum' else torch.max

        # Identify highlighting nodes:
        exp_nodes = self.enc_subgraph.nodes

        draw_args = dict()

        if norm_imps:
            # Normalize all importance scores:
            save_imps = [self.node_imp, self.edge_imp, self.feature_imp]
            save_imps = [s.clone() if s is not None else s for s in save_imps ]
            for s in (self.node_imp, self.edge_imp, self.feature_imp):
                if s is not None:
                    s = s / s.sum()

        if heat_by_prescence:
            if self.node_imp is not None:
                node_c = [int(i in exp_nodes) for i in subG.nodes]
                draw_args['node_color'] = node_c

        if heat_by_exp:
            if self.node_imp is not None:
                node_c = []
                for i in subG.nodes:
                    if i in self.enc_subgraph.nodes:
                        if isinstance(self.node_imp[self.node_reference[i]], torch.Tensor):
                            if self.node_imp[self.node_reference[i]].dim() > 0:
                                c = node_agg(self.node_imp[self.node_reference[i]]).item()
                            else:
                                c = self.node_imp[self.node_reference[i]].item()
                        else:
                            c = self.node_imp[self.node_reference[i]]
                    else:
                        c = 0

                    node_c.append(c)

                draw_args['node_color'] = node_c

            if self.edge_imp is not None:
                whole_edge_index, _ = subgraph(kadd_hop_neighborhood, edge_index = graph_data.edge_index)

                # Need to match edge indices across edge_index and edges in graph
                tuple_edge_index = [(whole_edge_index[0,i].item(), whole_edge_index[1,i].item()) \
                    for i in range(whole_edge_index.shape[1])]

                _, emask = remove_duplicate_edges(self.enc_subgraph.edge_index)
                # Remove self loops:
                emask_2 = torch.logical_not(self.enc_subgraph.edge_index[0,:] == self.enc_subgraph.edge_index[1,:])
                emask = emask & emask_2

                trimmed_enc_subg_edge_index = self.enc_subgraph.edge_index[:,emask]

                mask_edge_imp = self.edge_imp[emask].clone()

                # Find where edge_imp is applied on one duplicate edge but not another:
                masked_out_by_rmdup = self.enc_subgraph.edge_index[:,torch.logical_not(emask)]
                ones_in_rmdup = self.edge_imp[torch.logical_not(emask)].nonzero(as_tuple=True)[0]
                for j in ones_in_rmdup:
                    edge = masked_out_by_rmdup[:,j].tolist()
                    # Reverse the edge:
                    edge = edge[::-1] 
                    
                    trim_loc_mask = (trimmed_enc_subg_edge_index[0,:] == edge[0]) & (trimmed_enc_subg_edge_index[1,:] == edge[1])
                    trim_loc = (trim_loc_mask).nonzero(as_tuple=True)[0] 
                    if trim_loc.shape[0] > 0:
                        # Should be over 0 if we found it
                        trim_loc = trim_loc[0].item()
                        mask_edge_imp[trim_loc] = 1 # Ensure this edge is also one
                    # Don't do anything if we didn't find it

                positive_edge_indices = mask_edge_imp.nonzero(as_tuple=True)[0]

                # TODO: fix edge imp vis. to handle continuous edge importance scores
                mask_edge_imp = self.edge_imp[positive_edge_indices]

                positive_edges = [(trimmed_enc_subg_edge_index[0,e].item(), trimmed_enc_subg_edge_index[1,e].item()) \
                    for e in positive_edge_indices]

                # Tuples in list should be hashable
                edge_list = list(subG.edges)

                # Get dictionary with mapping from edge index to networkx graph
                #edge_matcher = match_torch_to_nx_edges(subG, remove_duplicate_edges(whole_edge_index)[0])
                edge_matcher = {edge_list[i]:i for i in range(len(edge_list))}
                for i in range(len(edge_list)):
                    forward_tup = edge_list[i]
                    backward_tup = tuple(list(edge_list[i])[::-1])
                    edge_matcher[forward_tup] = i
                    edge_matcher[backward_tup] = i

                edge_heat = torch.zeros(len(edge_list))
                #edge_heat = torch.zeros(whole_edge_index.shape[1])

                for e in positive_edges:
                    #e = positive_edges[i]
                    # Must find index, which is not very efficient
                    edge_heat[edge_matcher[e]] = 1

                draw_args['edge_color'] = edge_heat.tolist()
                #coolwarm cmap:
                draw_args['edge_cmap'] = plt.cm.coolwarm

            # Heat edge explanations if given

        # Seed the position to stay consistent:
        pos = nx.spring_layout(subG, seed = 1234)
        nx.draw(subG, pos, ax = ax, **draw_args, with_labels = show_node_labels)

        # Highlight the center node index:
        nx.draw(subG.subgraph(self.node_idx), pos, node_color = 'red', 
                node_size = 400, ax = ax)

        if norm_imps:
            self.node_imp, self.edge_imp, self.feature_imp = save_imps[0], save_imps[1], save_imps[2]

        if show:
            plt.show()

    def visualize_graph(self, ax = None, show = False, agg_nodes = torch.mean):
        '''
        Draws the graph of the Explanation

        '''

        if ax is None:
            ax = plt.gca()
        
        G = gxai_utils.to_networkx_conv(self.graph, to_undirected=True)

        draw_args = dict()

        # Node weights defined by node_imp:
        if self.node_imp is not None:
            # Get node weights
            #print('node imp shape', self.node_imp.shape)
            #print('num nodes', len(list(G.nodes())))
            if isinstance(self.node_imp, torch.Tensor):
                node_imp_heat = [agg_nodes(self.node_imp[n]).item() for n in G.nodes()]
                #node_imp_map = {i:self.node_imp[i].item() for i in range(G.number_of_nodes())}
            else:
                node_imp_heat = [agg_nodes(self.node_imp[n]) for n in G.nodes()]
                #node_imp_map = {i:self.node_imp[i] for i in range(G.number_of_nodes())}

            draw_args['node_color'] = node_imp_heat

        if self.edge_imp is not None:
            edge_matcher = match_torch_to_nx_edges(G, self.graph.edge_index)
            edge_heat = torch.zeros(G.number_of_edges())

            # for i in range(self.graph.edge_index.shape[1]):
            #     try:
            #         e1 = (self.graph.edge_index[0,i].item(), self.graph.edge_index[1,i].item())
            #         edge_heat[edge_matcher[e1]] = self.edge_imp[i].item()
            #     except KeyError:
            #         e2 = (self.graph.edge_index[1,i].item(), self.graph.edge_index[0,i].item())
            #         edge_heat[edge_matcher[e2]] = self.edge_imp[i].item()

            i = 0
            for e1, e2 in G.edges():
                edge_heat[i] = self.edge_imp[edge_matcher[(e1, e2)]].item()
                i += 1
                    
            draw_args['edge_color'] = edge_heat.tolist()
            draw_args['edge_cmap'] = plt.cm.coolwarm

        # Don't do anything for feature imp
        pos = nx.kamada_kawai_layout(G)
        nx.draw(G, pos, ax = ax, **draw_args)

        if show:
            plt.show()

        return G, pos


    def show_feature_imp(self, ax = None, show: bool = False):
        '''
        Show feature importance on a heatmap
        Args:
            ax (matplotlib axis, optional): Axis on which to draw the heatmap. 
                If :obj:`None`, heatmap is drawn on plt.gca(). 
                (:default: :obj:`None`)
            show (bool, optional): Whether to show the heatmap (:obj:`True`) 
                or not. (:default: :obj:`False`)
        
        No return
        '''

        ax = ax if ax is not None else plt.gca()

        # Draw a heatmap on the axis:
        feat_imp = self.feature_imp.numpy()

        ax.imshow(feat_imp.reshape((-1, 1)))

        if show:
            plt.show()

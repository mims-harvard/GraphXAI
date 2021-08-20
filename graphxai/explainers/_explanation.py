import torch

class Enclosing_Subgraph:
    def __init__(self, nodes, edge_index, inv, edge_mask, directed = False):
        self.nodes = nodes
        self.edge_index = edge_index
        self.inv = inv
        self.edge_mask = edge_mask
        self.directed = directed

class whole_graph:
    def __init__(self, x, edge_index, directed = False):
        self.x = x
        self.edge_index = edge_index
        self.directed = directed

class Explanation:
    '''
    Members:
        feature_imp (torch.Tensor): Feature importance scores
            - Size: (x1,) with x1 = number of features
        node_imp (torch.Tensor): Node importance scores
            - Size: (n,) with n = number of nodes in subgraph or graph
        edge_imp (torch.Tensor): Edge importance scores
            - Size: (e,) with e = number of edges in subgraph or graph
        graph (torch_geometric.data.Data): Original graph on which explanation
            was computed
            - Optional member, can be left None if graph is too large
    Optional members:
        enc_subgraph (Subgraph): k-hop subgraph around 
            - Corresponds to nodes and edges comprising computational graph around node
    '''
    def __init__(self):
        # Establish basic properties
        self.feature_imp = None
        self.node_imp = None
        self.edge_imp = None

        self.node_idx = None # Set this for node-level prediction explanations
        self.graph = None

    def set_enclosing_subgraph(self, k_hop_tuple):
        '''
        Args:
            k_hop_tuple (tuple): Return value from torch_geometric.utils.k_hop_subgraph
        '''
        self.enc_subgraph = Enclosing_Subgraph(*k_hop_tuple)

    def apply_subgraph_mask(self, mask_node = False, mask_edge = False):
        '''
        Performs automatic masking on the node and edge importance members
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
            self.node_imp = self.node_imp[self.subgraph]

    def set_whole_graph(self, x, edge_index):
        self.graph = whole_graph(x, edge_index)
import torch

class DemoExplainer:
    def __init__(self, model, criterion, *_):
        """
        Args:
            model (torch.nn.Module): model on which to make predictions
            criterion (torch.nn.Module): loss function
        """
        self.model = model
        self.criterion = criterion
        self.L = len([module for module in self.model.modules()
                      if isinstance(module, MessagePassing)])

    def get_explanation_node(self, node_idx: int, edge_index: torch.Tensor,
                             x: torch.Tensor, label: torch.Tensor, *_):
        """
        Explain a node prediction.

        Args:
            node_idx (int): index of the node to be explained
            edge_index (torch.Tensor, [2 x m]): edge index of the graph
            x (torch.Tensor, [n x d]): node features
            label (torch.Tensor, [n x ...]): labels to explain

        Returns:
            exp (dict):
                exp['feature'] (torch.Tensor, [d]): feature mask explanation
                exp['edge'] (torch.Tensor, [m]): k-hop edge mask explanation
            khop_info (4-tuple of torch.Tensor):
                0. the nodes involved in the subgraph
                1. the filtered `edge_index`
                2. the mapping from node indices in `node_idx` to their new location
                3. the `edge_index` mask indicating which edges were preserved
        """
        exp = {'feature': None, 'edge': None}

        num_hops = self.L
        khop_info = subset, sub_edge_index, mapping, _ = \
            k_hop_subgraph(node_idx, num_hops, edge_index,
                           relabel_nodes=True, num_nodes=x.shape[0])
        sub_x = x[subset]

        self.model.eval()
        # Compute explanation

        return exp, khop_info

    def get_explanation_graph(self, edge_index: torch.Tensor,
                              x: torch.Tensor, label: torch.Tensor, *_):
        """
        Explain a whole-graph prediction.

        Args:
            edge_index (torch.Tensor, [2 x m]): edge index of the graph
            x (torch.Tensor, [n x d]): node features
            label (torch.Tensor, [n x ...]): labels to explain
            forward_args (tuple, optional): additional arguments to model.forward

        Returns:
            exp (dict):
                exp['feature'] (torch.Tensor, [n x d]): feature mask explanation
                exp['edge'] (torch.Tensor, [m]): k-hop edge mask explanation
        """
        exp = {'feature': None, 'edge': None}

        self.model.eval()
        # Compute explanation

        return exp

    def get_explanation_link(self):
        """
        Explain an edge link prediction
        """
        raise NotImplementedError()

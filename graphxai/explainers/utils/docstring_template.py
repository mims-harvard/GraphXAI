'''
Explain one node prediction by the model.

Args:
    x (torch.Tensor): Tensor of node features from the entire graph
    node_idx (int): Node index for which to explain a prediction around
    edge_index (torch.Tensor): edge_index of entire graph
    label (int, optional): Label on which to compute the explanation for
        this node. If `None`, the predicted label from the model will be
        used. (default: :obj:`None`)
    forward_kwargs (dict, optional): Additional arguments to model.forward 
        beyond x and edge_index. Must be keyed on argument name. 
        (default: :obj:`{}`)

:rtype: :class:`graphxai.Explanation`

Returns:
    exp (:class:`Explanation`): Explanation output from the method.
        Fields are:
        `feature_imp`: :obj:`None`
        `node_imp`: :obj:`None`
        `edge_imp`: :obj:`None`
        `enc_subgraph`: :obj:`graphxai.utils.EnclosingSubgraph`
'''

'''
Explain one graph prediction by the model.

Args:
    x (torch.Tensor): Tensor of node features from the graph.
    edge_index (torch.Tensor): Edge_index of graph.
    label (int, optional): Label on which to compute the explanation for
        this node. If `None`, the predicted label from the model will be
        used. (default: :obj:`None`)
    forward_kwargs (dict, optional): Additional arguments to model.forward 
        beyond x and edge_index. Must be keyed on argument name. 
        (default: :obj:`{}`)

:rtype: :class:`graphxai.Explanation`

Returns:
    exp (:class:`Explanation`): Explanation output from the method. 
        Fields are:
        `feature_imp`: :obj:`None`
        `node_imp`: :obj:`None`
        `edge_imp`: :obj:`None`
        `graph`: :obj:`torch_geometric.data.Data`
'''
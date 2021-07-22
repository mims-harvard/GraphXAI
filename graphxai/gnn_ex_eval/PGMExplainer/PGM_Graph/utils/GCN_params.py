def net_params():
    
    n_heads = -1
    edge_feat = False
    pseudo_dim_MoNet = -1
    kernel = -1
    gnn_per_block = -1
    embedding_dim = -1
    pool_ratio = -1
    n_mlp_GIN = -1
    gated = False
    self_loop = False
    max_time = 48
    
    seed=41
    epochs=1000
    batch_size=5
    init_lr=5e-5
    lr_reduce_factor=0.5
    lr_schedule_patience=25
    min_lr = 1e-6
    weight_decay=0
    L=4
    hidden_dim=146
    out_dim=hidden_dim
    dropout=0.0
    readout='mean'
    
    net_params = {}
    net_params['device'] = None
    net_params['gated'] = gated  # for mlpnet baseline
    net_params['in_dim'] = 3
    net_params['in_dim_edge'] = 1
    net_params['residual'] = True
    net_params['hidden_dim'] = hidden_dim
    net_params['out_dim'] = out_dim
    num_classes = 10
    net_params['n_classes'] = num_classes
    net_params['n_heads'] = n_heads
    net_params['L'] = L  # min L should be 2
    net_params['readout'] = "sum"
    net_params['graph_norm'] = True
    net_params['batch_norm'] = True
    net_params['in_feat_dropout'] = 0.0
    net_params['dropout'] = 0.0
    net_params['edge_feat'] = edge_feat
    net_params['self_loop'] = self_loop
    net_params['pool_ratio'] = pool_ratio
    net_params['batch_size'] = batch_size
    
    return net_params
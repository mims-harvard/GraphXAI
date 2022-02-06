import torch
from graphxai.explainers import *

from graphxai.datasets import Mutagenicity, Benzene, FluorideCarbonyl
from graphxai.gnn_models.graph_classification import GIN_3layer, GCN_3layer

def get_exp_method(method, model, criterion, bah, node_idx, pred_class, data, device):
    method = method.lower()
    if method=='gnnex':
        raise ValueError('GNNEX does not support graph-level explanations')

    elif method=='grad':
        exp_method = GradExplainer(model, criterion = criterion)
        forward_kwargs={'x': data.x.to(device),
                        'y': data.y.to(device),
                        'node_idx': int(node_idx),
                        'edge_index': data.edge_index.to(device)}
    elif method=='gcam':
        exp_method = GradCAM(model, criterion = criterion)
        forward_kwargs={'x':data.x.to(device),
                        'y': data.y.to(device),
                        'node_idx': int(node_idx),
                        'edge_index': data.edge_index.to(device),
                        'average_variant': [True]}
    elif method=='gbp':
        exp_method = GuidedBP(model, criterion = criterion)
        forward_kwargs={'x': data.x.to(device),
                        'y': data.y.to(device),
                        'node_idx': int(node_idx),
                        'edge_index': data.edge_index.to(device)}
    elif method=='glime':
        raise ValueError('GLIME does not support graph-level explanations')

    elif method=='ig':
        exp_method = IntegratedGradExplainer(model, criterion = criterion)
        forward_kwargs = {'x': data.x.to(device),
                        'edge_index': data.edge_index.to(device),
                        'node_idx': int(node_idx),
                        'label': pred_class}
    elif method=='glrp':
        exp_method = GNN_LRP(model)
        forward_kwargs={'x': data.x.to(device),
                        'edge_index': data.edge_index.to(device),
                        'node_idx': node_idx,
                        'label': pred_class,
                        'edge_aggregator':torch.sum}
    elif method=='pgmex':
        exp_method=PGMExplainer(model, explain_graph=False, p_threshold=0.1)
        forward_kwargs={'node_idx': node_idx,
                        'x': data.x.to(device),
                        'edge_index': data.edge_index.to(device),
                        'top_k_nodes': 10}
                        
    elif method=='pgex':
        raise ValueError('PGEX does not support graph-level explanations')

    elif method=='rand':
        exp_method = RandomExplainer(model)
        forward_kwargs={'x': data.x.to(device),
                        'node_idx': int(node_idx),
                        'edge_index': data.edge_index.to(device)}
    elif method=='subx':
        exp_method = SubgraphX(model, reward_method = 'gnn_score', num_hops = bah.model_layers)
        forward_kwargs={'node_idx': node_idx,
                        'x': data.x.to(device),
                        'edge_index': data.edge_index.to(device),
                        'label': pred_class,
                        'max_nodes': 15}
    else:
        OSError('Invalid argument!!')

    # Add dummy batch to forward_kwargs:
    forward_kwargs['forward_kwargs'] = {'batch': torch.tensor([0]).long().to(device)}

    return exp_method, forward_kwargs

def get_model(name):

    # if dataset_name.lower() ==  'mutagenicity':
    #     ifeat = 14
    # elif (dataset_name.lower() == 'benzene') or (dataset_name.lower() == 'fluoridecarbonyl'):
    #     ifeat = 14
    # else:
    #     OSError('Invalid dataset name!')

    # All datasets have same numbers of input_feat: 14

    if name.lower() == 'gcn':
        model = GCN_3layer(32, input_feat = 14, classes = 2)
    elif name.lower() == 'gin':
        model = GIN_3layer(32, input_feat = 14, classes = 2)
    else:
        OSError('Invalid model!')
    return model
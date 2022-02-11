import os
import torch
from graphxai.explainers import *

from graphxai.datasets import Mutagenicity, Benzene, FluorideCarbonyl
from graphxai.gnn_models.graph_classification import GIN_3layer, GCN_3layer, GAT_3layer, SAGE_3layer, JKNet_3layer

def get_exp_method(method, model, criterion, pred_class, data, device, train_pg = False, emb_layer_name='conv3'):
    method = method.lower()
    if method=='gnnex':
        #raise ValueError('GNNEX does not support graph-level explanations')
        exp_method = GNNExplainer(model)
        forward_kwargs={'x': data.x.to(device),
                        'edge_index': data.edge_index.to(device)}

    elif method=='grad':
        exp_method = GradExplainer(model, criterion = criterion)
        forward_kwargs={'x': data.x.to(device),
                        'label': data.y.to(device),
                        'edge_index': data.edge_index.to(device)}

    elif method == 'cam':
        exp_method = CAM(model, activation = lambda x: torch.argmax(x, dim=1))
        forward_kwargs={'x':data.x.to(device),
                        'label': data.y.to(device),
                        'edge_index': data.edge_index.to(device)}

    elif method=='gcam':
        exp_method = GradCAM(model, criterion = criterion)
        forward_kwargs={'x':data.x.to(device),
                        'label': data.y.to(device),
                        'edge_index': data.edge_index.to(device),
                        'average_variant': True}

    elif method=='gbp':
        exp_method = GuidedBP(model, criterion = criterion)
        forward_kwargs={'x': data.x.to(device),
                        'y': data.y.to(device),
                        'edge_index': data.edge_index.to(device)}

    elif method=='glime':
        raise ValueError('GLIME does not support graph-level explanations')

    elif method=='ig':
        exp_method = IntegratedGradExplainer(model, criterion = criterion)
        forward_kwargs = {'x': data.x.to(device),
                        'edge_index': data.edge_index.to(device),
                        'label': pred_class}

    elif method=='pgmex':
        exp_method=PGMExplainer(model, explain_graph=True, p_threshold=0.1)
        forward_kwargs={'x': data.x.to(device),
                        'edge_index': data.edge_index.to(device),
                        'top_k_nodes': None}

    elif method=='pgex':
        exp_method = PGExplainer(model, explain_graph = True, emb_layer_name=emb_layer_name, max_epochs=10, lr=0.1)
        forward_kwargs = {
            'x': data.x.to(device),
            'edge_index': data.edge_index.to(device),
            'label': pred_class
        }
        if train_pg:
            exp_method.train_explanation_model(data.to(device))
        #raise ValueError('PGEX does not support graph-level explanations')

    elif method=='rand':
        exp_method = RandomExplainer(model)
        forward_kwargs={'x': data.x.to(device),
                        'edge_index': data.edge_index.to(device)}

    elif method=='subx':
        exp_method = SubgraphX(model, reward_method = 'gnn_score', num_hops = 3)
        forward_kwargs={'x': data.x.to(device),
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
        model = GCN_3layer(hidden_channels=32, in_channels = 14, out_channels=2)
    elif name.lower() == 'gin':
        model = GIN_3layer(hidden_channels=32, in_channels = 14, out_channels=2)
    elif name.lower() == 'sage':
        model = SAGE_3layer(hidden_channels=32, in_channels = 14, out_channels=2)
    elif name.lower() == 'gat':
        model = GAT_3layer(hidden_channels=32, in_channels = 14, out_channels=2)
    elif name.lower() == 'jknet':
        model = JKNet_3layer(hidden_channels=32, in_channels = 14, out_channels=2)
    else:
        OSError('Invalid model!')
    return model

default_root = '~/GraphXAI/formal/real_world'
def get_dataset(name, root = default_root, device = None):
    '''
    Args:
        root: Only needed if using Mutagenicity
    '''

    if name.lower() == 'benzene':
        return Benzene(split_sizes = (0.7,0.15,0.15), seed = 1234, device = device)

    elif (name.lower() == 'fc') or (name.lower() == 'fluoridecarbonyl'):
        return FluorideCarbonyl(split_sizes = (0.7, 0.15, 0.15), seed = 1234, device = device)

    elif (name.lower() == 'mutagenicity') or (name.lower() == 'mutag'): 
        return Mutagenicity(root = os.path.join(root, 'data'), split_sizes = (0.7, 0.15, 0.15), seed = 1234, device = device)

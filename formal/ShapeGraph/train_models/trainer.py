import tqdm
import ipdb
import argparse, sys; sys.path.append('../..')
import random as rand
import torch

from graphxai.gnn_models.node_classification.testing import GIN_3layer_basic, GCN_3layer_basic, GSAGE_3layer

def get_model(name):
    if name.lower() == 'gcn':
        model = GCN_3layer_basic(16, input_feat = 11, classes = 2)
    elif name.lower() == 'gin':
        model = GIN_3layer_basic(16, input_feat = 11, classes = 2)
    elif name.lower() == 'sage':
        # Get SAGE model
        model = GSAGE_3layer(16, input_feat = 11, classes = 2)
    else:
        OSError('Invalid model!')
    return model

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help = 'Name of model to train (GIN, GCN, or SAGE)')
parser.add_argument('--dataset', required=True, help = 'Location of ShapeGraph dataset on which to train the model')
parser.add_argument('--save_dir', default='./trained/', help='Directory to which to send trained models.')
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = get_model(name = args.model).to(device)
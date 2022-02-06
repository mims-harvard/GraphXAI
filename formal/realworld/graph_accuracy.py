import os, sys
import argparse, random
import numpy as np
import torch

from utils import get_model, get_exp_method, get_dataset

from graphxai.datasets import Benzene
from graphxai.metrics.metrics_graph import graph_exp_acc_graph, graph_exp_faith_graph

parser = argparse.ArgumentParser()
parser.add_argument('--exp_method', required=True, help='name of the explanation method')
parser.add_argument('--dataset', required=True, help = 'benzene, mutag, or FC')
parser.add_argument('--faithfulness', action='store_true')
parser.add_argument('--accuracy', action='store_true')
parser.add_argument('--model', required = True, default='GIN', type=str, help = 'name of model for benchmarking')
parser.add_argument('--save_dir', default='./fairness_results/', help='folder for saving results')
args = parser.parse_args()

args.dataset = args.dataset.lower()

assert args.dataset in ['benzene', 'mutag', 'fc'], "Dataset must be benzene, mutag, or FC" 

seed_value=912
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Get dataset:
dataset = get_dataset(args.dataset, device = device)
test_inds = dataset.test_index # Index of testing objects

# Set up model:
model = get_model(args.model)

# Load model:
# Construct path to model:
mpath = os.path.join(args.dataset, 'model_weights', '{}_{}.pth'.format(args.model.upper(), args.dataset))
model.load_state_dict(torch.load(mpath))

criterion = torch.nn.CrossEntropyLoss().to(device)

for idx in test_inds:
    # Gets graph:
    data = dataset.graphs[idx].to(device)

    pred = model(data.x, data.edge_index, batch = torch.zeros((1,)).long()).reshape(-1, 1).argmax(dim=0)

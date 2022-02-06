import os, sys
import argparse, random
import numpy as np
import torch

from utils import get_model, get_exp_method, get_dataset

from graphxai.datasets import Benzene
from graphxai.utils.performance.load_exp import exp_exists_graph
from graphxai.metrics.metrics_graph import graph_exp_acc_graph, graph_exp_faith_graph

my_base_graphxai = '~/GraphXAI/formal/realworld'

parser = argparse.ArgumentParser()
parser.add_argument('--exp_method', required=True, help='name of the explanation method')
parser.add_argument('--dataset', required=True, help = 'benzene, mutag, or FC')
parser.add_argument('--faithfulness', action='store_true')
parser.add_argument('--accuracy', action='store_true')
parser.add_argument('--model', required = True, default='GIN', type=str, help = 'name of model for benchmarking')
#parser.add_argument('--save_dir', default='./fairness_results/', help='folder for saving results')
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

gea_node = []
gea_edge = []

gef_node = []
gef_edge = []

GEA_PATH = os.path.join(args.dataset, 'results', 'accuracy')
GEF_PATH = os.path.join(args.dataset, 'results', 'faithfulness')

add_args = {
    'batch': torch.zeros((1,)).long()
}

for idx in test_inds:
    # Gets graph, ground truth explanation:
    data, gt_exp = dataset[idx]

    data = data.to(device)

    pred_class = model(data.x, data.edge_index, **add_args).reshape(-1, 1).argmax(dim=0)

    if pred_class != data.y.item():
        continue

    # Try to load from existing dir:
    exp = exp_exists_graph(idx, path = os.path.join(my_base_graphxai, args.dataset, 'EXPS', args.exp_method.upper()))

    if exp is None:
        explainer, forward_kwargs = get_exp_method(args.exp_method, model, criterion, pred_class, data, device)
        exp = explainer.get_explanation_node(**forward_kwargs)

    if args.accuracy:
        # Accuracy:
        _, node_acc, edge_acc = graph_exp_acc_graph(gt_exp, exp)

        gea_node.append(node_acc)
        gea_edge.append(edge_acc)

    if args.faithfulness:
        # Faithfulness:
        _, node_faith, edge_faith = graph_exp_faith_graph(exp, dataset, model, forward_kwargs = add_args)

        gef_node.append(node_faith)
        gef_edge.append(edge_faith)

# Save:
if args.accuracy:
    np.save(open(os.path.join(GEA_PATH, f'{args.exp_method}_GEA_node.npy'), 'wb'), gea_node)
    np.save(open(os.path.join(GEA_PATH, f'{args.exp_method}_GEA_edge.npy'), 'wb'), gea_edge)

if args.faithfulness:
    np.save(open(os.path.join(GEF_PATH, f'{args.exp_method}_GEF_node.npy'), 'wb'), gef_node)
    np.save(open(os.path.join(GEF_PATH, f'{args.exp_method}_GEF_edge.npy'), 'wb'), gef_edge)
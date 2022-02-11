import os, sys; sys.path.append('..')
import argparse
import torch
import numpy as np
from tqdm import trange
from torch_geometric.datasets import TUDataset

from graphxai.explainers import IntegratedGradExplainer
#from graphxai.explainers.utils.visualizations import visualize_mol_explanation
from graphxai.gnn_models.graph_classification import train, test
from graphxai.gnn_models.graph_classification.gcn import GCN_2layer, GCN_3layer
from graphxai.gnn_models.graph_classification.gin import GIN_2layer, GIN_3layer
from graphxai.datasets import Benzene

from utils import get_dataset, get_model

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help = 'benzene, mutag, or FC')
parser.add_argument('--model', required = True, default='GIN', type=str, help = 'name of model for benchmarking')
#parser.add_argument('--save_dir', default='./fairness_results/', help='folder for saving results')
args = parser.parse_args()

# Load data: ------------------------------------------
#dataset = Benzene(split_sizes = (0.8, 0.2, 0), seed = seed)
dataset = get_dataset(args.dataset.lower(), device = device)
train_loader, _ = dataset.get_train_loader(batch_size = 64)
test_loader, _ = dataset.get_test_loader()
val_loader, _ = dataset.get_val_loader()

# Train GNN model -------------------------------------
n_trials = 10
overall_auroc = []

mfile_name = os.path.join('{}_{}.pth'.format(args.model.upper(), args.dataset.lower()))

for i in range(n_trials):
    model = get_model(args.model.lower()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    best_f1 = 0

    for epoch in trange(1, 101):
        train(model, optimizer, criterion, train_loader)
        #f1, prec, rec, auprc, auroc = test(model, test_loader)
        f1, prec, rec, auprc, auroc = test(model, val_loader)

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), mfile_name)

        #print(f'Epoch: {epoch:03d}, Test F1: {f1:.4f}, Test AUROC: {auroc:.4f}')

    model.load_state_dict(torch.load(mfile_name))
    f1, precision, recall, auprc, auroc = test(model, test_loader)
    print(f'Test F1: {f1:.4f}, Test AUROC: {auroc:.4f}')
    overall_auroc.append(auroc)

print('')
print('Average AUROC: {} +- {}'.format(np.mean(overall_auroc), np.std(overall_auroc) / np.sqrt(len(overall_auroc))))
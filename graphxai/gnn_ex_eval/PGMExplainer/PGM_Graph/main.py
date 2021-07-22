import numpy as np
from scipy.special import softmax
import torch
from torch.utils.data import DataLoader
import networkx as nx
from torchvision import datasets as ds
from torchvision import transforms
import argparse

from nets.superpixels_graph_classification.load_net import gnn_model # import all GNNS
from data.data import LoadData # import dataset
from utils import GCN_params

import pgm_explainer_graph as pe # import explainer

def arg_parse():
    parser = argparse.ArgumentParser(description="PGM Explainer arguments.")
    parser.add_argument(
            "--start", dest="start", type=int, help="Index of starting image."
        )
    parser.add_argument(
            "--end", dest="end", type=int, help="Index of ending image."
        )
    parser.add_argument("--perturb-indicator", dest="perturb_indicator", help="diff or abs.")
    parser.add_argument("--perturb-mode", dest="perturb_mode", help="mean, zero, max or uniform.")
    parser.add_argument("--perturb-feature", dest="perturb_feature", help="color or location.")
    
    parser.set_defaults(
        start = 0,
        end = 1,
        perturb_indicator = "diff",
        perturb_mode = "mean",
        perturb_feature = "color"
    )
    return parser.parse_args()

prog_args = arg_parse()

MNIST_test_dataset = ds.MNIST(root='PATH', train=False, download=True, transform=transforms.ToTensor())
MODEL_NAME = 'GCN'
DATASET_NAME = 'MNIST'
dataset = LoadData(DATASET_NAME)
trainset, valset, testset = dataset.train, dataset.val, dataset.test

net_params = GCN_params.net_params()
model = gnn_model(MODEL_NAME, net_params)
model.load_state_dict(torch.load("data/superpixels/epoch_188.pkl"))
model.eval()

test_loader = DataLoader(testset, batch_size=1, shuffle=False, drop_last=False, collate_fn=dataset.collate)

index_to_explain = range(prog_args.start, prog_args.end)
if prog_args.perturb_feature == "color":
    perturb_features_list = [0]
elif prog_args.perturb_feature == "location":
    perturb_features_list = [1,2]


Explanations = []
for iter, (graph, label, snorm_n, snorm_e) in enumerate(test_loader):
    if iter in index_to_explain:
        pred = model.forward(graph, graph.ndata['feat'],graph.edata['feat'],snorm_n, snorm_e)
        soft_pred = np.asarray(softmax(np.asarray(pred[0].data)))
        pred_threshold = 0.1*np.max(soft_pred)
        e = pe.Graph_Explainer(model, graph, 
                               snorm_n = snorm_n, snorm_e = snorm_n, 
                               perturb_feature_list = perturb_features_list,
                               perturb_mode = prog_args.perturb_mode,
                               perturb_indicator = prog_args.perturb_indicator)
        pgm_nodes, p_values, candidates = e.explain(num_samples = 1000, percentage = 10, 
                                top_node = 4, p_threshold = 0.05, pred_threshold = pred_threshold)
        label = np.argmax(soft_pred)
        pgm_nodes_filter = [i for i in pgm_nodes if p_values[i] < 0.02]
        x_cor = [e.X_feat[node_][1] for node_ in pgm_nodes_filter]
        y_cor = [e.X_feat[node_][2] for node_ in pgm_nodes_filter]
        result = [iter, label, pgm_nodes_filter, x_cor, y_cor]
        print(result)
        Explanations.append(result)
        savedir = 'result/explanations_'+ str(prog_args.start) + "_" + str(prog_args.end) +".txt"
        with open(savedir, "a") as text_file:
            text_file.write(str(result) + "\n")
            
            
            
            
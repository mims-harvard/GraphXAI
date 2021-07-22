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

import utils.shap_bg as shap_utils

def arg_parse():
    parser = argparse.ArgumentParser(description="PGM Explainer arguments.")
    parser.add_argument(
            "--start", dest="start", type=int, help="Index of starting image."
        )
    parser.add_argument(
            "--end", dest="end", type=int, help="Index of ending image."
        )
    
    parser.set_defaults(
        start = 0,
        end = 1
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

train_loader = DataLoader(trainset, batch_size=1, shuffle=False, drop_last=False, collate_fn=dataset.collate)
test_loader = DataLoader(testset, batch_size=1, shuffle=False, drop_last=False, collate_fn=dataset.collate)
            
index_to_explain = range(prog_args.start, prog_args.end)
mnist_bg = shap_utils.get_background()
offset = 0.001
top_node = 3
Explanations = []
for iter, (graph, label, snorm_n, snorm_e) in enumerate(test_loader):
    if iter in index_to_explain:  
        features = graph.ndata['feat'].requires_grad_()
        pred = model.forward(graph, graph.ndata['feat'],graph.edata['feat'],snorm_n, snorm_e)
        soft_pred = np.asarray(softmax(np.asarray(pred[0].data)))
        label = np.argmax(soft_pred)
        grad = torch.autograd.grad(pred[0,label], features)[0]
        phi0 = torch.zeros(features.shape[0])
        for j in range(features.shape[0]):
            phi0[j] = grad[j,0]*(features[j,0] - 
                                 shap_utils.pos_to_color(features[j,2].detach().item(),1-features[j,1].detach().item(),mnist_bg) 
                                + offset)
#         abs_shap = np.abs(phi0.data.numpy())
#         shap_nodes = list(np.argpartition(abs_shap, -top_node)[-top_node:])
        shap_value = phi0.data.numpy()
        shap_nodes = list(np.argpartition(shap_value, -top_node)[-top_node:])
        x_cor = [features[node_].data.numpy()[1] for node_ in shap_nodes]
        y_cor = [features[node_].data.numpy()[2] for node_ in shap_nodes]
        result = [iter, label, shap_nodes, x_cor, y_cor]
        print(result)
        Explanations.append(result)
        
        savedir = 'result/shap_explanations_'+ str(prog_args.start) + "_" + str(prog_args.end) +".txt"
        with open(savedir, "a") as text_file:
            text_file.write(str(result) + "\n")
            
            
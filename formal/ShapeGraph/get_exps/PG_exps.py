import tqdm
import ipdb
import os, argparse, sys
import random as rand
import torch
import sys; sys.path.append('/home/owq978/GraphXAI/formal')
from metrics import *
from graphxai.explainers import *
from graphxai.datasets  import load_ShapeGraph
from graphxai.datasets.shape_graph import ShapeGraph
from graphxai.utils.performance.load_exp import exp_exists
from graphxai.gnn_models.node_classification.testing import GIN_3layer_basic

my_base_graphxai = '/home/owq978/GraphXAI'

bah = torch.load(open('/home/owq978/GraphXAI/data/ShapeGraph/unzipped/SG_homophilic.pickle', 'rb'))

data = bah.get_graph(use_fixed_split=True)

test_set = torch.load(os.path.join(my_base_graphxai, 'formal', 'ShapeGraph', 'test_inds_SG_homophilic.pt'))
# Test on 3-layer basic GCN, 16 hidden dim:
model = GIN_3layer_basic(16, input_feat = 11, classes = 2).to(device)
mpath = os.path.join(my_base_graphxai, 'formal/model_weights/model_homophily.pth')
model.load_state_dict(torch.load(mpath))

pred = model(data.x.to(device), data.edge_index.to(device)) # Model predictions

# Train entirety of PGExplainer:
explainer = PGExplainer(model, emb_layer_name = 'gin3' if isinstance(model, GIN_3layer_basic) else 'gcn3', max_epochs=10, lr=0.1)
explainer.train_explanation_model(data.to(device))
torch.save(explainer, open(os.path.join(my_base_graphxai, 'formal/ShapeGraph/get_exps', 'PGExplainer.pt'), 'wb'))

save_exp_dir = os.path.join(my_base_graphxai, 'formal/ShapeGraph', 'bigSG_explanations', 'PGEX')

for node_idx in tqdm.tqdm(test_set):
    node_idx = node_idx.item()

    # Get predictions
    pred_class = pred[node_idx, :].reshape(-1, 1).argmax(dim=0)

    if pred_class != data.y[node_idx]:
        # Don't evaluate if the prediction is incorrect
        continue

    forward_kwargs={'node_idx': node_idx,
                'x': data.x.to(device),
                'edge_index': data.edge_index.to(device),
                'label': pred_class}

    #exp = explainer.get_explanation_node(**forward_kwargs)

    # Get explanations
    # exp = exp_exists(node_idx, path = save_exp_dir, get_exp = False) # Retrieve the explanation, if it's there
    # #print(exp)

    # if not exp:
    exp = explainer.get_explanation_node(**forward_kwargs)
    torch.save(exp, open(os.path.join(save_exp_dir, 'exp_node{:0<5d}.pt'.format(node_idx)), 'wb'))

# Pickle the whole object:
#torch.save(explainer, open(os.path.join(my_base_graphxai, 'formal/ShapeGraph/get_exps', 'PGExplainer.pt'), 'wb'))

import argparse
import argparse, sys; sys.path.append('../..')
import random as rand
import torch
from metrics import *
import numpy as np
from graphxai.gnn_models.node_classification.testing import GIN_3layer_basic, GCN_3layer_basic, GSAGE_3layer

# ----------------------------
my_base_graphxai = '/home/owq978/GraphXAI'
data_path = 'data/ShapeGraph/unzipped/SG_homophilic.pickle'

model_name = 'gin'
model_path = 'formal/model_weights/model_homophily.pth'

delta_save_path = 'formal/model_weights/model_homophily_delta.npy'
# ----------------------------

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

device = "cuda" if torch.cuda.is_available() else "cpu"

bah = torch.load(open(os.path.join(my_base_graphxai, data_path), 'rb'))
data = bah.get_graph(use_fixed_split=True)

model = get_model(model_name).to(device)
model.load_state_dict(torch.load(os.path.join(my_base_graphxai, model_path)))

delta = calculate_delta(data.x.to(device), data.edge_index.to(device), torch.where(data.train_mask == True)[0], model = model, label=data.y, sens_idx=[bah.sensitive_feature], device = device)
np.save(open(os.path.join(my_base_graphxai, delta_save_path), 'wb'), np.array([delta]))

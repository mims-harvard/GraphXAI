import tqdm
import argparse, sys, os
import random as rand
import torch
from graphxai.explainers import *
from graphxai.datasets  import load_ShapeGraph
from graphxai.datasets.shape_graph import ShapeGraph
from graphxai.utils.performance.load_exp import exp_exists
from graphxai.gnn_models.node_classification.testing import GIN_3layer_basic

my_base_graphxai = '/home/owq978/GraphXAI'
sys.path.append(os.path.join(my_base_graphxai, 'formal'))
from metrics import *

parser = argparse.ArgumentParser()
parser.add_argument('--start_ind', required=True, type=int, help='Can be negative or positive')
args = parser.parse_args()

start_ind = args.start_ind

bah = torch.load(open('/home/owq978/GraphXAI/data/ShapeGraph/unzipped/SG_homophilic.pickle', 'rb'))

data = bah.get_graph(use_fixed_split=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Test set remains static
test_set = torch.load(os.path.join(my_base_graphxai, 'formal', 'ShapeGraph', 'test_inds_SG_homophilic.pt')).numpy()
# Test on 3-layer basic GCN, 16 hidden dim:
model = GIN_3layer_basic(16, input_feat = 11, classes = 2).to(device)
mpath = os.path.join(my_base_graphxai, 'formal/model_weights/model_homophily.pth')
model.load_state_dict(torch.load(mpath))

pred = model(data.x.to(device), data.edge_index.to(device)) # Model predictions

explainer = SubgraphX(model, reward_method = 'gnn_score', num_hops = bah.model_layers)

if start_ind < 0:
    range_gen = range(len(test_set) - 1 + start_ind, len(test_set) // 2 - 4, -3)
else:
    range_gen = range(start_ind, len(test_set) // 2 + 4, 3)

save_exp_dir = os.path.join(my_base_graphxai, 'formal/ShapeGraph', 'bigSG_explanations', 'SUBX')

for i in tqdm.tqdm(range_gen):

    #node_idx = torch.tensor(test_set[i]).to(device)
    node_idx = test_set[i]

    pred_class = pred[node_idx, :].reshape(-1, 1).argmax(dim=0) 

    if exp_exists(node_idx, path = save_exp_dir, get_exp = False) or (pred_class != data.y[node_idx]):
        continue

    forward_kwargs={'node_idx': node_idx,
                    'x': data.x.to(device),
                    'edge_index': data.edge_index.to(device),
                    'label': pred_class,
                    'max_nodes': 15}

    exp = explainer.get_explanation_node(**forward_kwargs)
    torch.save(exp, open(os.path.join(save_exp_dir, 'exp_node{:0<5d}.pt'.format(node_idx)), 'wb'))

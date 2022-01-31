import tqdm
import ipdb
import argparse, sys
import random as rand
import torch
from metrics import *
from graphxai.explainers import *
from graphxai.datasets  import load_ShapeGraph
from graphxai.datasets.shape_graph import ShapeGraph
from graphxai.gnn_models.node_classification.testing import GIN_3layer_basic


def get_exp_method(method, model, criterion, bah):
    if method=='gnnex':
        exp_method = GNNExplainer(model)
    elif method=='grad':
        exp_method = GradExplainer(model, criterion = criterion)
        forward_kwargs={'x': data.x.to(device),
                        'y': data.y.to(device),
                        'node_idx': int(node_idx),
                        'edge_index': data.edge_index.to(device)}
    elif method=='gcam':
        exp_method = GradCAM(model, criterion = criterion)
        forward_kwargs={'x':data.x.to(device),
                        'y': data.y.to(device),
                        'node_idx': int(node_idx),
                        'edge_index' = data.edge_iandex.to(device),
                        'average_variant'=[True]}
    elif method=='gbp':
        exp_method = GuidedBP(model, criterion = criterion)
        forward_kwargs={'x': data.x.to(device),
                        'y': data.y.to(device),
                        'node_idx': int(node_idx),
                        'edge_index': data.edge_index.to(device)}
    elif method=='glime':
        exp_method = GraphLIME(model)
        forward_kwargs={'x': data.x.to(device),
                        'node_idx': int(node_idx),
                        'edge_index': data.edge_index.to(device)}
    elif method=='ig':
        exp_method = IntegratedGradExplainer(model, criterion = criterion)
        forward_kwargs = {'x': data.x.to(device),
                        'edge_index': data.edge_index.to(device),
                        'node_idx': node_idx,
                        'label': pred_class}
    elif method=='glrp':
        exp_method = GNN_LRP(model)
        forward_kwargs={'x': data.x.to(device),
                        'edge_index': data.edge_index.to(device),
                        'node_idx': node_idx,
                        'label': pred_class,
                        'edge_aggregator':torch.sum}
    elif method=='pgmex':
        exp_method=PGMExplainer(model, explain_graph=False, p_threshold=0.1)
        forward_kwargs={'node_idx': node_idx,
                        'x': 'data.x',
                        'edge_index': data.edge_index,
                        'top_k_nodes' = 10}
    elif method=='pgex':
        exp_method=PGExplainer(modeal, emb_layer_name = 'gin3' if isinstance(model, GIN_3layer_basic) else 'gcn3', max_epochs=10, lr=0.1)
        exp_method.train_explanation_model(bah.get_graph(use_fixed_split=True))
    elif method=='rand':
        exp_method = RandomExplainer(model)
        forward_kwargs={'x': data.x.to(device),
                        'node_idx': int(node_idx),
                        'edge_index': data.edge_index.to(device)}
    elif method=='subx':
        exp_method = SubgraphX(model, reward_method = 'gnn_score', num_hops = bah.model_layers)
    else:
        OSError('Invalid argument!!')
    return exp_method


parser = argparse.ArgumentParser()
parser.add_argument('--exp_method', required=True, help='name of the explanation method')
parser.add_argument('--save_dir', default='./results_homophily/', help='folder for saving results')
args = parser.parse_args()

seed_value=912
rand.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load ShapeGraph dataset
# Smaller graph is shown to work well with model accuracy, graph properties
bah = torch.load(open('/home/cha567/GraphXAI/data/ShapeGraph/SG_homophilic.pickle', 'rb'))

data = bah.get_graph(use_fixed_split=True)

inhouse = (data.y[data.test_mask] == 1).nonzero(as_tuple=True)[0]
np.random.shuffle(inhouse.numpy())
print(inhouse)
# Test on 3-layer basic GCN, 16 hidden dim:
model = GIN_3layer_basic(16, input_feat = 11, classes = 2).to(device)

# Get prediction of a node in the 2-house class:
model.load_state_dict(torch.load('model_homophily.pth'))
# model.eval()

gef_feat = []
gef_node = []
gef_edge = []

# Get predictions
pred = model(data.x.to(device), data.edge_index.to(device))

criterion = torch.nn.CrossEntropyLoss().to(device)
explainer = get_exp_method(args.exp_method, model, criterion, bah)

for node_idx in tqdm.tqdm(inhouse[:1000]):
    # print(node_idx)
    # node_idx = rand.choice(inhouse).item()
    node_idx = node_idx.item()

    pred_class = pred[node_idx, :].reshape(-1, 1).argmax(dim=0).item()

    # print('GROUND TRUTH LABEL: \t {}'.format(data.y[node_idx].item()))
    # print('PREDICTED LABEL  : \t {}'.format(pred.argmax(dim=0).item()))

    ipdb.set_trace()
    exp = explainer.get_explanation_node(**forward_kwargs)
    exp = explainer.get_explanation_node(
                        x = data.x.to(device),
                        y = data.y.to(device),
                        node_idx = int(node_idx),
                        edge_index = data.edge_index.to(device))
    feat, node, edge = graph_exp_faith(exp, bah, model, sens_idx=[bah.sensitive_feature])

    gef_feat.append(feat)
    gef_node.append(node)
    gef_edge.append(edge)


############################
# Saving the metric values
# save_dir='./results_homophily/'
np.save(f'{args.save_dir}{args.exp_method}_gef_feat.npy', gef_feat)
np.save(f'{args.save_dir}{args.exp_method}_gef_node.npy', gef_node)
np.save(f'{args.save_dir}{args.exp_method}_gef_edge.npy', gef_edge)

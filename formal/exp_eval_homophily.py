import ipdb
import random as rand
import torch
from metrics import *
from graphxai.explainers import *
from graphxai.datasets  import load_ShapeGraph
from graphxai.datasets.shape_graph import ShapeGraph
from graphxai.gnn_models.node_classification.testing import GIN_3layer_basic


# Load ShapeGraph dataset
# Smaller graph is shown to work well with model accuracy, graph properties
bah = torch.load(open('/home/cha567/GraphXAI/data/ShapeGraph/SG_homophilic.pickle', 'rb'))

data = bah.get_graph(use_fixed_split=True)
inhouse = (data.y == 1).nonzero(as_tuple=True)[0]
np.random.shuffle(inhouse.numpy())

# Test on 3-layer basic GCN, 16 hidden dim:
model = GIN_3layer_basic(16, input_feat = 11, classes = 2)

# Get prediction of a node in the 2-house class:
model.load_state_dict(torch.load('model_homophily.pth'))
# model.eval()

gef_feat = []
gef_node = []
gef_edge = []

# gnnexp = GNNExplainer(model)
pred = model(data.x, data.edge_index)

criterion = torch.nn.CrossEntropyLoss()
grad = GradExplainer(model, criterion = criterion)

for node_idx in inhouse[:100]:
    print(node_idx)
    # node_idx = rand.choice(inhouse).item()
    node_idx = node_idx.item()

    pred_class = pred[node_idx, :].reshape(-1, 1).argmax(dim=0).item()

    # print('GROUND TRUTH LABEL: \t {}'.format(data.y[node_idx].item()))
    # print('PREDICTED LABEL  : \t {}'.format(pred.argmax(dim=0).item()))

    # exp = gnnexp.get_explanation_node(
    #                     node_idx = node_idx,
    #                     x = data.x.cuda(),
    #                     edge_index = data.edge_index.cuda())
    exp = grad.get_explanation_node(
                        x = data.x,
                        node_idx = int(node_idx),
                        edge_index = data.edge_index)
    ipdb.set_trace()
    feat, node, edge = graph_exp_faith(exp, bah, model, sens_idx=[bah.sensitive_feature])

    gef_feat.append(feat)
    gef_node.append(node)
    gef_edge.append(edge)

ipdb.set_trace()

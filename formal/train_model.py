import ipdb
import random as rand
import torch
import argparse
import numpy as np
from graphxai.gnn_models.node_classification.testing import GCN_3layer_basic, GIN_3layer_basic, test, train, val
from graphxai.datasets.shape_graph import ShapeGraph
from graphxai.datasets  import load_ShapeGraph


parser = argparse.ArgumentParser()
parser.add_argument('--expt_name', required=True, help='name of the explanation method')
args = parser.parse_args()


# Load ShapeGraph dataset
if args.expt_name == 'homophily':
    ipdb.set_trace()
    bah = torch.load(open('/home/cha567/GraphXAI/data/ShapeGraph/SG_small_homophily.pickle', 'rb'))
elif args.expt_name == 'heterophily':
    bah = torch.load(open('/home/cha567/GraphXAI/data/ShapeGraph/SG_small_heterophily.pickle', 'rb'))
elif args.expt_name == 'triangle':
    bah = torch.load(open('/home/cha567/GraphXAI/data/ShapeGraph/SG_triangles.pickle', 'rb'))
elif args.expt_name == 'fair':
    bah = torch.load(open('/home/cha567/GraphXAI/data/ShapeGraph/SG_fair.pickle', 'rb'))
else:
    print('Invalid Input!!')
    exit(0)

# Fix seed
seed_value=123123
rand.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)

data = bah.get_graph(use_fixed_split=True)

# Test on 3-layer basic GCN, 16 hidden dim:
model = GIN_3layer_basic(16, input_feat = 11, classes = 2)
# model.load_state_dict(torch.load(f'./model_SG_org_heto.pth'))
# f1, acc, precision, recall, auroc, auprc = test(model, data, get_auc=True)
# print(f'Test F1: {f1:.4f}, Test AUROC: {auroc:.4f}')
# exit(0)

# Train the model:
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

best_f1=0
for epoch in range(0, 18):
    loss = train(model, optimizer, criterion, data)
    f1, acc, precision, recall, auroc, auprc = val(model, data, get_auc=True)
    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), f'./model_weights/model_small_{args.expt_name}.pth')
    if epoch % 1 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val F1: {f1:.4f}, Val AUROC: {auroc:.4f}')

# Testing performance
f1, acc, precision, recall, auprc, auroc = test(model, data, get_auc=True)
print(f'Test F1: {f1:.4f}, Test AUROC: {auroc:.4f}')

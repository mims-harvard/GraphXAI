import ipdb
import random as rand
import torch
import argparse
import numpy as np
from graphxai.gnn_models.node_classification.testing import GCN_3layer_basic, GIN_3layer_basic, test, train, val, GSAGE_3layer, GAT_3layer_basic, JKNet_3layer
from graphxai.datasets.shape_graph import ShapeGraph
from graphxai.datasets  import load_ShapeGraph


parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='name of the gnn model to be trained')
args = parser.parse_args()


bah = torch.load(open('/home/cha567/GraphXAI/data/ShapeGraph/SG_homophily.pickle', 'rb'))

# Fix seed
# seed_value=1
# rand.seed(seed_value)
# np.random.seed(seed_value)
# torch.manual_seed(seed_value)

data = bah.get_graph(use_fixed_split=True)

f1_list = []
auroc_list = []
for _ in range(10):

    if args.model == 'gcn':
        # Test on 3-layer basic GCN, 16 hidden dim:
        model = GCN_3layer_basic(16, input_feat = 11, classes = 2)
    elif args.model == 'gin':
        model = GIN_3layer_basic(16, input_feat = 11, classes = 2)
    elif args.model == 'sage':
        model = GSAGE_3layer(16, input_feat = 11, classes = 2)
    elif args.model == 'gat':
        model = GAT_3layer_basic(16, input_feat = 11, classes = 2)
    elif args.model == 'jk':
        model = JKNet_3layer(16, input_feat = 11, classes = 2)
    else:
        OSError('Wrong model input!!')

    # Train the model:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    best_f1=0
    for epoch in range(0, 1001):
        loss = train(model, optimizer, criterion, data)
        f1, acc, precision, recall, auroc, auprc = val(model, data, get_auc=True)
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), f'./model_weights/{args.model}_benchmark.pth')
        # if epoch % 1 == 0:
        #     print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val F1: {f1:.4f}, Val AUROC: {auroc:.4f}')

    # Testing performance
    f1, acc, precision, recall, auroc, auprc = test(model, data, get_auc=True)
    print(f'Test F1: {f1:.4f}, Test AUROC: {auroc:.4f}')
    f1_list.append(f1)
    auroc_list.append(auroc)

print (f'Average Test F1: {np.mean(f1_list)}+-{np.std(f1_list)} | AUROC: {np.mean(auroc_list)}+-{np.std(auroc_list)}')

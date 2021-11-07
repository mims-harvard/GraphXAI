import sys
import torch

import numpy as np

from graphxai.datasets import BAShapes
from graphxai.gnn_models.node_classification.testing import *

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

# assert len(sys.argv) == 4, 'usage: python3 test_BAH_vis.py <shape_insert_strategy> <labeling_method> <feature_method>'

# assert sys.argv[1] in ['ub', 'lb', 'global'], "shape_insert_strategy must be in ['ub', 'lb', 'global']"
# assert sys.argv[2] in ['e', 'f', 'ef'], "labeling_method must be in ['e', 'f', 'ef']"
# assert sys.argv[3] in ['gaussian', 'ns', 'onehot'], "feature_method must be in ['gaussian', 'ns', 'onehot']"

# pm_conv = {'ub':'neighborhood upper bound', 'lb':'local', 'global':'global'}
# lm_conv = {'e':'edge', 'f':'feature', 'ef':'edge and feature'}
# fm_conv = {'gaussian':'gaussian', 'ns':'network stats', 'onehot':'onehot'}

# if sys.argv[1] == 'global':
#     num_shapes = 5
# elif sys.argv[1] == 'lb':
#     num_shapes = 1
# elif sys.argv[1] == 'ub':
#     num_shapes = None

# class Hyperparameters:
#     num_hops = 1
#     n = 1000
#     m = 1
#     num_shapes = None
#     shape_insert_strategy = 'neighborhood upper bound'
#     shape_upper_bound = 1
#     labeling_method = 'edge'


class Hyperparameters:
    num_hops = 2
    n = 30 # Invalid for bound
    m = 1 # Invalid for bound
    num_shapes = 5 # Invalid
    shape_insert_strategy = 'neighborhood upper bound'
    shape_upper_bound = 1
    shape = 'random'
    labeling_method = 'edge'
    feature_method = 'gaussian_lv'

if __name__ == '__main__':
    hyp = Hyperparameters
    args = {key:value for key, value in hyp.__dict__.items() if not key.startswith('__') and not callable(value)}

    n_trials = 10

    avg_accs = []
    for n in range(n_trials):

        bah = BAShapes(**args)
        data = bah.get_graph(use_fixed_split=True, split_sizes = [0.7, 0.3, 0])

        #model = GCN_3layer(64, input_feat=3, classes=4)
        model = GCN_2layer(64, input_feat=10, classes=4)
        #model = GIN_1layer(input_feat=10, classes=4)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()

        all_accs = []
        for epoch in range(1,1000):
            loss = train(model, optimizer, criterion, data)
            acc = test(model, data, num_classes = 4)
            all_accs.append(acc)
            #print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Acc: {acc:.4f}')
        avg_accs.append(max(all_accs))
        

    for i in range(4):
        print(f'Count {i}:', (data.y == i).nonzero(as_tuple=True)[0].shape[0])
    #print('Count 1:', (data.y == 1).nonzero(as_tuple=True)[0].shape[0])
    print('Avg. Max', np.mean(avg_accs))
    #print('Epochs:', np.argmax(all_accs))
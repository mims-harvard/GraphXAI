import sys
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from graphxai.datasets.BA_shapes.ba_houses import BAHousesRandomGaussFeatures,\
    BAHousesRandomOneHotFeatures
from graphxai.datasets import BAShapes

from graphxai.gnn_models.node_classification.testing import *
# Contains GCN 1-3 layers, GIN 1-3 layers, train and test

latent_dim = 64

assert len(sys.argv) == 4, \
    "usage: python3 random_feat_predict.py <'gaussian', 'onehot' or 'network stats'> <GIN or GCN> <# layers (1-3)>"

class Hyperparameters:
    num_hops = 1
    n = 50
    m = 1
    num_shapes = None
    shape_insert_strategy = 'neighborhood upper bound'
    shape_upper_bound = 1
    labeling_method = 'edge'

hyp = Hyperparameters

if sys.argv[1] not in ['network_stats', 'gaussian', 'onehot']:
    raise NotImplementedError("Please input Gauss or OneHot for feature label strategy")

args = {key:value for key, value in hyp.__dict__.items() if not key.startswith('__') and not callable(value)}

bah = BAShapes(**args, feature_method = sys.argv[1])

# if sys.argv[1] == 'Gauss':
#     bah = BAHousesRandomGaussFeatures(
#     num_hops=1,
#     n=5000,
#     m=1,
#     num_houses=None, # Get max number possible
#     shape_insert_strategy='neighborhood upper bound',
#     shape_upper_bound = 1)

# elif sys.argv[1] == 'OneHot':
#     # bah = BAHousesRandomOneHotFeatures(
#     # num_hops=1,
#     # n=5000,
#     # m=1,
#     # num_houses=None,
#     # shape_insert_strategy='neighborhood upper bound',
#     # shape_upper_bound = 1)


# else:
#     raise NotImplementedError("Please input Gauss or OneHot for feature label strategy")

input_features = int(bah.graph.x.shape[1])
# print(input_features)

if sys.argv[2] == 'GCN':
    if int(sys.argv[3]) == 1:
        model = GCN_1layer(input_feat = input_features, classes = 2)

    if int(sys.argv[3]) == 2:
        model = GCN_2layer(latent_dim, input_feat = input_features, classes = 2)

    if int(sys.argv[3]) == 3:
        model = GCN_3layer(latent_dim, input_feat = input_features, classes = 2)

elif sys.argv[2] == 'GIN':
    if int(sys.argv[3]) == 1:
        model = GIN_1layer(input_feat = input_features, classes = 2)

    if int(sys.argv[3]) == 2:
        model = GIN_2layer(latent_dim, input_feat = input_features, classes = 2)

    if int(sys.argv[3]) == 3:
        model = GIN_3layer(latent_dim, input_feat = input_features, classes = 2)

else:
    raise NotImplementedError("Please input GCN or GIN for model")

data = bah.get_graph(use_fixed_split=False, split_sizes = [0.7, 0.3, 0.0])
print(data.x.shape)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

class_numbers = [(data.y == i).nonzero(as_tuple=True)[0].shape[0] for i in [1, 0]]
weights = torch.tensor([max(class_numbers) / cn for cn in class_numbers])

criterion = torch.nn.CrossEntropyLoss(weight=weights)

all_f1s = []
all_acs = []
all_prec = []
all_rec = []
for epoch in range(1,400):
    loss = train(model, optimizer, criterion, data)
    #print('Loss', loss.item())
    f1, acc, prec, rec = test(model, data)
    all_f1s.append(f1)
    all_acs.append(acc)
    all_prec.append(prec)
    all_rec.append(rec)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test F1: {f1:.4f}, Test Acc: {acc:.4f}, P: {prec:.4f}, R: {rec:.4f}')

print('Count 0:', (data.y == 0).nonzero(as_tuple=True)[0].shape[0])
print('Count 1:', (data.y == 1).nonzero(as_tuple=True)[0].shape[0])
print('Max', max(all_f1s))
print('Epochs:', np.argmax(all_f1s))

x = list(range(len(all_f1s)))
plt.plot(x, all_f1s, label = 'F1')
plt.plot(x, all_acs, label = 'Accuracy')
plt.plot(x, all_prec, label = 'Precision')
plt.plot(x, all_rec, label = 'Recall')
plt.title('Metrics on {} ({} layers), {} Features'.format(sys.argv[2], sys.argv[3], sys.argv[1]))
plt.xlabel('Epochs')
plt.ylabel('Metric')
plt.legend()
plt.show()
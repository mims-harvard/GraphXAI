import ipdb
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from numpy import ndarray
from torch_geometric.utils import to_networkx
from graphxai.datasets.shape_graph import ShapeGraph
from graphxai.utils import to_networkx_conv, Explanation, distance
from tqdm import trange

from test_homophily import homophily_test


bah = ShapeGraph(model_layers=3, seed=912, make_explanations=True, num_subgraphs=500, prob_connection=0.0075, subgraph_size=9, class_sep=0.5, n_informative=6, verify=True)

# Get the graph information from ShapeGraph
feature = bah.get_graph().x.clone().detach()
label = bah.get_graph().y
edge_index = bah.get_graph().edge_index
#ipdb.set_trace()

# Get indicecs for informative features
#ind = torch.where( bah.feature_imp_true.long() == 1)[0]
# ind = ind[ind!=bah.sensitive_feature]

ind = (bah.feature_imp_true.long() == 1)

# Initialize the optimizer
optimizer = torch.optim.Adam([feature], lr=0.3)
feature.requires_grad = True
opt_feature = feature.clone()


# Optimizaton parameters
num_epochs=50
homophily_coeff=1.0
sample = 10

# Get indices for connected nodes having same label
c_inds = torch.randperm(edge_index.shape[1])[:sample]
c_inds = c_inds[label[edge_index.t()[c_inds][:, 0]] == label[edge_index.t()[c_inds][:, 1]]]

# Get indices for connected nodes having different label
nc_inds = torch.randperm(edge_index.shape[1])[:sample]
nc_inds = nc_inds[label[edge_index.t()[nc_inds][:, 0]] != label[edge_index.t()[nc_inds][:, 1]]]

for i in trange(num_epochs):
    # Compute similarities for all edges in the c_inds:
    c_cos_sim = F.cosine_similarity(feature[edge_index.t()[c_inds][:, 0]], feature[edge_index.t()[c_inds][:, 1]])
    nc_cos_sim = F.cosine_similarity(feature[edge_index.t()[nc_inds][:, 0]], feature[edge_index.t()[nc_inds][:, 1]])
    optimizer.zero_grad()
    loss = -homophily_coeff * c_cos_sim.sum() + (1-homophily_coeff)*nc_cos_sim.sum()
    loss.backward()
    optimizer.step()
  
# Only perturb the non-informative attribute and change the informative ones to original values
feature[:, ind] = opt_feature[:, ind]

#homophily_test()
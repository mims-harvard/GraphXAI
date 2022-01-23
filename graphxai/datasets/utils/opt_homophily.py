import torch
import torch.nn.functional as F


# bah = ShapeGraph(model_layers=3, seed=912, make_explanations=True, num_subgraphs=500, prob_connection=0.0075, subgraph_size=9, class_sep=0.5, n_informative=6, verify=True)

# # Get the graph information from ShapeGraph
# label = bah.get_graph().y.detach().clone()
# edge_index = bah.get_graph().edge_index
# #ipdb.set_trace()

# # Get indicecs for informative features
# #ind = torch.where( bah.feature_imp_true.long() == 1)[0]
# # ind = ind[ind!=bah.sensitive_feature]

# #feature_ind = bah.feature_imp_true

# # Get mask over features that we need to optimize:
# opt_mask = torch.logical_not(bah.feature_imp_true)
# opt_mask[bah.sensitive_feature] = False

# to_opt = bah.get_graph().x.detach().clone()[opt_mask]

# # Initialize the optimizer
# # optimizer = torch.optim.Adam([feature], lr=0.3)
# # feature.requires_grad = True
# # opt_feature = feature.clone()

# # Initialize optimizer:
# optimizer = torch.optim.Adam([to_opt], lr=0.3)
# to_opt.requires_grad = True


# # Optimizaton parameters
# num_epochs=50
# homophily_coeff=1.0
# sample = 10

# # Get indices for connected nodes having same label
# c_inds = torch.randperm(edge_index.shape[1])[:sample]
# c_inds = c_inds[label[edge_index.t()[c_inds][:, 0]] == label[edge_index.t()[c_inds][:, 1]]]

# # Get indices for connected nodes having different label
# nc_inds = torch.randperm(edge_index.shape[1])[:sample]
# nc_inds = nc_inds[label[edge_index.t()[nc_inds][:, 0]] != label[edge_index.t()[nc_inds][:, 1]]]

# for i in trange(num_epochs):
#     # Compute similarities for all edges in the c_inds:
#     c_cos_sim = F.cosine_similarity(to_opt[edge_index.t()[c_inds][:, 0]], to_opt[edge_index.t()[c_inds][:, 1]])
#     nc_cos_sim = F.cosine_similarity(to_opt[edge_index.t()[nc_inds][:, 0]], to_opt[edge_index.t()[nc_inds][:, 1]])
#     optimizer.zero_grad()
#     loss = -homophily_coeff * c_cos_sim.sum() + (1-homophily_coeff)*nc_cos_sim.sum()
#     loss.backward()
#     optimizer.step()
  
# Only perturb the non-informative attribute and change the informative ones to original values
#feature[:, ind] = opt_feature[:, ind]

def optimize_homophily(
        x, 
        edge_index,
        label,
        feature_mask, 
        homophily_coef = 1.0, 
        epochs = 50, 
        batch_size = 10
    ):

    # opt_mask = torch.logical_not(feature_mask)
    # opt_mask[bah.sensitive_feature] = False

    to_opt = x.detach().clone()[:,feature_mask]

    optimizer = torch.optim.Adam([to_opt], lr=0.3)
    to_opt.requires_grad = True

    # Get indices for connected nodes having same label
    c_inds = torch.randperm(edge_index.shape[1])[:batch_size]
    c_inds = c_inds[label[edge_index.t()[c_inds][:, 0]] == label[edge_index.t()[c_inds][:, 1]]]

    # Get indices for connected nodes having different label
    nc_inds = torch.randperm(edge_index.shape[1])[:batch_size]
    nc_inds = nc_inds[label[edge_index.t()[nc_inds][:, 0]] != label[edge_index.t()[nc_inds][:, 1]]]

    for i in range(epochs):
        # Compute similarities for all edges in the c_inds:
        c_cos_sim = F.cosine_similarity(to_opt[edge_index.t()[c_inds][:, 0]], to_opt[edge_index.t()[c_inds][:, 1]])
        nc_cos_sim = F.cosine_similarity(to_opt[edge_index.t()[nc_inds][:, 0]], to_opt[edge_index.t()[nc_inds][:, 1]])
        optimizer.zero_grad()
        #loss = -homophily_coef * c_cos_sim.sum() + (1-homophily_coef)*nc_cos_sim.sum()
        loss = -homophily_coef * c_cos_sim.sum() + (homophily_coef - 1)*nc_cos_sim.sum()
        loss.backward()
        optimizer.step()

    # Assign to appropriate copies:
    xcopy = x.detach().clone()
    xcopy[:,feature_mask] = to_opt

    return xcopy.contiguous()
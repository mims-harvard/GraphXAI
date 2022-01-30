import ipdb
import random
import torch
from graphxai.gnn_models.node_classification.testing import GCN_3layer_basic, GIN_3layer_basic, test, train 
from graphxai.datasets.shape_graph import ShapeGraph
from graphxai.datasets  import load_ShapeGraph

# Load ShapeGraph dataset
# Smaller graph is shown to work well with model accuracy, graph properties
bah = torch.load(open('/home/cha567/GraphXAI/data/ShapeGraph/SG_homophilic.pickle', 'rb'))

#bah = ShapeGraph(model_layers = 3, 
#    make_explanations=True,
#    num_subgraphs = 1200, 
#    prob_connection = 0.0075, 
#    subgraph_size = 11,
#    class_sep=0.3,
#    n_informative=4,
#    homophily_coef=1,
#    n_clusters_per_class=1,
#    seed=912,
#    verify=True)

data = bah.get_graph(use_fixed_split=True)
inhouse = (data.y == 1).nonzero(as_tuple=True)[0]

# Test on 3-layer basic GCN, 16 hidden dim:
model = GIN_3layer_basic(16, input_feat = 11, classes = 2)

# Train the model:
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

best_auroc=0
for epoch in range(1, 101):
    loss = train(model, optimizer, criterion, data)
    f1, acc, precision, recall, auroc, auprc = test(model, data, get_auc = True)
    if auroc > best_auroc:
        best_auroc = auroc
        torch.save(model.state_dict(), 'model_homophily.pth')

    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test F1: {f1:.4f}, Test AUROC: {auroc:.4f}')

ipdb.set_trace()



# Get prediction of a node in the 2-house class:
model.load_state_dict(torch.load('model_homophily.pth'))
model.eval()
node_idx = random.choice(inhouse).item()
pred = model(data.x, data.edge_index)[node_idx, :].reshape(-1, 1)
pred_class = pred.argmax(dim=0).item()

print('GROUND TRUTH LABEL: \t {}'.format(data.y[node_idx].item()))
print('PREDICTED LABEL   : \t {}'.format(pred.argmax(dim=0).item()))

# Run Explainer ----------------------------------------------------------
gnnexp = GNNExplainer(model)
exp = gnnexp.get_explanation_node(
                    node_idx = node_idx, 
                    x = data.x,  
                    edge_index = data.edge_index)
# ------------------------------------------------------------------------


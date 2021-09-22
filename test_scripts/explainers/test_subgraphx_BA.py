import random
import torch
import matplotlib.pyplot as plt

from graphxai.gnn_models.node_classification import BA_Houses as BAH
from graphxai.gnn_models.node_classification import GCN
from graphxai.explainers.utils.visualizations import *
from graphxai.explainers.utils import whole_graph_mask_to_subgraph
from graphxai.explainers.subgraphx import SubgraphX

def get_data(n, m, num_houses):
    bah = BAH(n, m)
    BAG = bah.make_BA_shapes(num_houses)
    data = bah.make_data(BAG, multiple_features = True)
    inhouse = bah.in_house
    return data, list(inhouse)

n = 500
m = 2
num_houses = 20
data, inhouse = get_data(n, m, num_houses)

model = GCN(64, input_feat = 3, classes = 2)
L = 2

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

def train():
      model.train()
      optimizer.zero_grad()  # Clear gradients.
      out = model(data.x, data.edge_index)  # Perform a single forward pass.
      loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
      loss.backward()  # Derive gradients.
      optimizer.step()  # Update parameters based on gradients.
      return loss

def test():
      model.eval()
      out = model(data.x, data.edge_index)
      pred = out.argmax(dim=1)  # Use the class with highest probability.
      test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
      return test_acc

for epoch in range(1, 201):
    loss = train()
    acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Acc: {acc:.4f}')


node_idx = int(random.choice(inhouse))

model.eval()
pred = model(data.x, data.edge_index)[node_idx,:].view(-1, 1)#.reshape(-1, 1)
pred_class = pred.argmax(dim=0).item()

explainer = SubgraphX(model, reward_method = 'gnn_score')

exp = explainer.get_explanation_node(data.x, label = int(pred_class), edge_index = data.edge_index, node_idx = node_idx, max_nodes = 10)

subgraph_node_mask, _ = whole_graph_mask_to_subgraph(node_mask = exp.node_imp, subgraph_nodes = exp.enc_subgraph.nodes)

fig, axs = plt.subplots(1, 2)

axs[0].set_title('k-hop subgraph')
visualize_subgraph_explanation(exp.enc_subgraph.edge_index, node_idx = node_idx, ax = axs[0], show = False, connected = True)


axs[1].set_title('SubgraphX Chosen Subgraph')
visualize_subgraph_explanation(exp.enc_subgraph.edge_index, node_idx = node_idx, node_weights = subgraph_node_mask.type(torch.int32),
    ax = axs[1], show = False, connected = False)

ymin, ymax = axs[0].get_ylim()
xmin, xmax = axs[0].get_xlim()
axs[0].text(xmin, ymax - 0.1*(ymax-ymin), 'Label = {:d}'.format(1))
axs[0].text(xmin, ymax - 0.15*(ymax-ymin), 'Pred  = {:d}'.format(pred_class))


plt.show()

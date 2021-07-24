import random
import torch
import matplotlib.pyplot as plt

from graphxai.explainers.cam import CAM, Grad_CAM
from graphxai.explainers.utils.testing_datasets import BA_Houses
from graphxai.explainers.utils.visualizations import visualize_subgraph_explanation
from graphxai.gnns import GCN
from graphxai.gnns.utils import train, test


n = 300
m = 2
num_houses = 20

bah = BA_Houses(n, m)
data, inhouse = bah.get_data(num_houses)

model = GCN(64, input_feat = 1, classes = 2)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(1, 201):
    loss = train(model, optimizer, criterion, data)
    acc = test(model, data)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Acc: {acc:.4f}')

node_idx = random.choice(inhouse)

model.eval()
pred = model(data.x, data.edge_index)[node_idx, :].reshape(-1, 1)
print('pred shape', pred.shape)

print('GROUND TRUTH LABEL: \t {}'.format(data.y[node_idx].item()))
print('PREDICTED LABEL   : \t {}'.format(pred.argmax(dim=0).item()))

act = lambda x: torch.argmax(x, dim=1)
cam = CAM(model, '', activation = act)

exp, khop_info = cam.get_explanation_node(data.x, node_idx = int(node_idx), edge_index = data.edge_index)
subgraph_eidx = khop_info[1]

#visualize_categorical_graph(data, show = True)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

# Ground truth plot:
#gt = [khop_info[0][i] for i in len(khop_info[0].tolist())]
print('Number house nodes:', sum(data.y.tolist()))
visualize_subgraph_explanation(subgraph_eidx, data.y.tolist(), node_idx = int(node_idx), ax = ax1, show = False)
ax1.set_title('Ground Truth')

# CAM plot:
visualize_subgraph_explanation(subgraph_eidx, exp, node_idx = int(node_idx), ax = ax2, show = False)
ax2.set_title('CAM')

gcam = Grad_CAM(model, '', criterion = criterion)
exp, khop_info = gcam.get_explanation_node(data.x, data.y, int(node_idx), data.edge_index, average_variant=False)

#Grad-CAM plot:
visualize_subgraph_explanation(subgraph_eidx, exp, node_idx = int(node_idx), ax = ax3, show = False)
ax3.set_title('Grad-CAM')

ymin, ymax = ax1.get_ylim()
xmin, xmax = ax1.get_xlim()
ax1.text(xmin, ymax - 0.1*(ymax-ymin), 'Label = {:d}'.format(data.y[node_idx].item()))
ax1.text(xmin, ymax - 0.15*(ymax-ymin), 'Pred  = {:d}'.format(pred.argmax(dim=0).item()))

plt.show()

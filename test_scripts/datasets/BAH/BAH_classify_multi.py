import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from graphxai.datasets.ba_houses import BAHouses as BAH
#from graphxai.gnn_models.node_classification import train, test

class GCN_1layer(torch.nn.Module):
    def __init__(self, hidden_channels, input_feat, classes):
        super(GCN_2layer, self).__init__()
        self.conv1 = GCNConv(input_feat, hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return x

class GCN_2layer(torch.nn.Module):
    def __init__(self, hidden_channels, input_feat, classes):
        super(GCN_2layer, self).__init__()
        self.conv1 = GCNConv(input_feat, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def train(model, optimizer,
          criterion, data):
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(data.x, data.edge_index)  # Perform a single forward pass.
    # print('Out shape', out.shape)
    # print('y shape', data.y.shape)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss
    
def test(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
    return test_acc

# Opts:
class BAH_opts:
    n = 1000
    m = 2
    k = 1
    seed = None
    shape_insert_strategy = 'local'

params = BAH_opts()

bah = BAH(
        num_hops = 2, 
        n = params.n,
        m = params.m,
        k = params.k,
        seed = params.seed,
        shape_insert_strategy = params.shape_insert_strategy,
        in_hood_numbering = True)

data = bah.get_graph()
num_classes = data.y.unique().max().item()
print(num_classes)

model = GCN_2layer(128, input_feat=3, classes=num_classes + 1)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(1,1000):
    loss = train(model, optimizer, criterion, data)
    acc = test(model, data)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Acc: {acc:.4f}')
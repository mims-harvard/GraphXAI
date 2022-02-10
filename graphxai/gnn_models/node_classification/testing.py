import ipdb
import torch
from torch_geometric.nn import GCNConv, GINConv, BatchNorm, SAGEConv, JumpingKnowledge, GATConv
from torch_geometric.nn import Sequential

import sklearn.metrics as metrics
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

class GCN_1layer(torch.nn.Module):
    def __init__(self, input_feat, classes):
        super(GCN_1layer, self).__init__()
        self.conv1 = GCNConv(input_feat, classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return x

class GCN_2layer(torch.nn.Module):
    def __init__(self, hidden_channels, input_feat, classes):
        super(GCN_2layer, self).__init__()
        self.conv1 = GCNConv(input_feat, hidden_channels)
        self.batchnorm1 = BatchNorm(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.batchnorm1(x)
        x = x.relu()
        x = self.conv2(x, edge_index)
        return x

class GCN_3layer(torch.nn.Module):
    def __init__(self, hidden_channels, input_feat, classes):
        super(GCN_3layer, self).__init__()
        self.gcn1 = GCNConv(input_feat, hidden_channels)
        self.batchnorm1 = BatchNorm(hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, hidden_channels)
        self.batchnorm2 = BatchNorm(hidden_channels)
        self.gcn3 = GCNConv(hidden_channels, classes)

    def forward(self, x, edge_index):
        x = self.gcn1(x, edge_index)
        x = self.batchnorm1(x)
        x = x.relu()
        x = self.gcn2(x, edge_index)
        x = self.batchnorm2(x)
        x = x.relu()
        x = self.gcn3(x, edge_index)
        return x

class GCN_3layer_basic(torch.nn.Module):
    def __init__(self, hidden_channels, input_feat, classes):
        super(GCN_3layer_basic, self).__init__()
        self.gcn1 = GCNConv(input_feat, hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, hidden_channels)
        self.gcn3 = GCNConv(hidden_channels, classes)

    def forward(self, x, edge_index):
        x = self.gcn1(x, edge_index)
        x = x.relu()
        x = self.gcn2(x, edge_index)
        x = x.relu()
        x = self.gcn3(x, edge_index)
        return x

class GCN_4layer_basic(torch.nn.Module):
    def __init__(self, hidden_channels, input_feat, classes):
        super(GCN_4layer_basic, self).__init__()
        self.gcn1 = GCNConv(input_feat, hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, hidden_channels)
        self.gcn3 = GCNConv(hidden_channels, hidden_channels)
        self.gcn4 = GCNConv(hidden_channels, classes)

    def forward(self, x, edge_index):
        x = self.gcn1(x, edge_index)
        x = x.relu()
        x = self.gcn2(x, edge_index)
        x = x.relu()
        x = self.gcn3(x, edge_index)
        x = x.relu()
        x = self.gcn4(x, edge_index)

        return x

# ----------------------- GINs ----------------------------

class GIN_1layer(torch.nn.Module):
    def __init__(self, input_feat, classes):
        super(GIN_1layer, self).__init__()
        self.mlp_gin1 = torch.nn.Linear(input_feat, classes)
        # Use gin3 naming for convention in PGEx
        self.gin3 = GINConv(self.mlp_gin1)

    def forward(self, x, edge_index):
        x = self.gin3(x, edge_index)
        return x

class GIN_2layer(torch.nn.Module):
    def __init__(self, hidden_channels, input_feat, classes):
        super(GIN_2layer, self).__init__()
        self.mlp_gin1 = torch.nn.Linear(input_feat, hidden_channels)
        self.gin1 = GINConv(self.mlp_gin1)
        self.mlp_gin2 = torch.nn.Linear(hidden_channels, classes)
        self.gin3 = GINConv(self.mlp_gin2)

    def forward(self, x, edge_index):
        x = self.gin1(x, edge_index)
        x = x.relu()
        x = self.gin3(x, edge_index)
        return x

class GIN_3layer(torch.nn.Module):
    def __init__(self, hidden_channels, input_feat, classes):
        super(GIN_3layer, self).__init__()
        self.mlp_gin1 = torch.nn.Linear(input_feat, hidden_channels)
        self.gin1 = GINConv(self.mlp_gin1)
        self.batchnorm1 = BatchNorm(hidden_channels)
        self.mlp_gin2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.gin2 = GINConv(self.mlp_gin2)
        self.batchnorm2 = BatchNorm(hidden_channels)
        self.mlp_gin3 = torch.nn.Linear(hidden_channels, classes)
        self.gin3 = GINConv(self.mlp_gin3)

    def forward(self, x, edge_index):
        x = self.gin1(x, edge_index)
        x = self.batchnorm1(x)
        x = x.relu()
        x = self.gin2(x, edge_index)
        x = self.batchnorm2(x)
        x = x.relu()
        x = self.gin3(x, edge_index)
        return x


class GIN_3layer_basic(torch.nn.Module):
    def __init__(self, hidden_channels, input_feat, classes):
        super(GIN_3layer_basic, self).__init__()
        self.mlp_gin1 = torch.nn.Linear(input_feat, hidden_channels)
        self.gin1 = GINConv(self.mlp_gin1)
        # self.batchnorm1 = BatchNorm(hidden_channels)
        self.mlp_gin2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.gin2 = GINConv(self.mlp_gin2)
        # self.batchnorm2 = BatchNorm(hidden_channels)
        self.mlp_gin3 = torch.nn.Linear(hidden_channels, classes)
        self.gin3 = GINConv(self.mlp_gin3)

    def forward(self, x, edge_index):
        x = self.gin1(x, edge_index)
        #x = self.batchnorm1(x)
        x = x.relu()
        x = self.gin2(x, edge_index)
        #x = self.batchnorm2(x)
        x = x.relu()
        x = self.gin3(x, edge_index)
        return x

class GIN_4layer_basic(torch.nn.Module):
    def __init__(self, hidden_channels, input_feat, classes):
        super(GIN_4layer_basic, self).__init__()
        self.mlp_gin1 = torch.nn.Linear(input_feat, hidden_channels)
        self.gin1 = GINConv(self.mlp_gin1)
        self.mlp_gin2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.gin2 = GINConv(self.mlp_gin2)
        self.mlp_gin3 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.gin3 = GINConv(self.mlp_gin3)
        self.mlp_gin4 = torch.nn.Linear(hidden_channels, classes)
        self.gin4 = GINConv(self.mlp_gin4)

    def forward(self, x, edge_index):
        x = self.gin1(x, edge_index)
        x = x.relu()
        x = self.gin2(x, edge_index)
        x = x.relu()
        x = self.gin3(x, edge_index)
        x = x.relu()
        x = self.gin4(x, edge_index)
        return x

# ----------------------- GraphSAGEs ----------------------------
class GSAGE_3layer(torch.nn.Module):
    def __init__(self, hidden_channels, input_feat, classes):
        super(GSAGE_3layer, self).__init__()
        self.gsage1 = SAGEConv(input_feat, hidden_channels, normalize = True)
        self.batchnorm1 = BatchNorm(hidden_channels)
        self.gsage2 = SAGEConv(hidden_channels, hidden_channels, normalize = True)
        self.batchnorm2 = BatchNorm(hidden_channels)
        self.gsage3 = SAGEConv(hidden_channels, classes)

    def forward(self, x, edge_index):
        x = self.gsage1(x, edge_index)
        x = self.batchnorm1(x)
        x = x.relu()
        x = self.gsage2(x, edge_index)
        x = self.batchnorm2(x)
        x = x.relu()
        x = self.gsage3(x, edge_index)
        return x

# ----------------------- JKNets ----------------------------
class JKNet_3layer(torch.nn.Module):
    def __init__(self, hidden_channels, input_feat, classes):
        super(JKNet_3layer, self).__init__()
        self.jknet = Sequential( 'x, edge_index', [
            (SAGEConv(input_feat, hidden_channels), 'x, edge_index -> x1'),
            (BatchNorm(hidden_channels), 'x1 -> x1'),
            (torch.nn.PReLU(), 'x1 -> x1'),
            (SAGEConv(hidden_channels, hidden_channels), 'x1, edge_index -> x2'),
            (BatchNorm(hidden_channels), 'x2 -> x2'),
            (torch.nn.PReLU(), 'x2 -> x2'),
            (SAGEConv(hidden_channels, hidden_channels), 'x2, edge_index -> x3'),
            (BatchNorm(hidden_channels), 'x3 -> x3'),
            (torch.nn.PReLU(), 'x3 -> x3'),
            (lambda x1, x2, x3: [x1, x2, x3], 'x1, x2, x3 -> xs'),
            (JumpingKnowledge('cat', hidden_channels, num_layers = 2), 'xs -> x'),
            torch.nn.Linear(3 * hidden_channels, classes),
        ]
        )

    def forward(self, x, edge_index):
        return self.jknet(x, edge_index)

class JKNet_3layer_lstm(torch.nn.Module):
    def __init__(self, hidden_channels, input_feat, classes):
        super(JKNet_3layer_lstm, self).__init__()
        self.jknet = Sequential( 'x, edge_index', [
            (GATConv(input_feat, hidden_channels), 'x, edge_index -> x1'),
            (BatchNorm(hidden_channels), 'x1 -> x1'),
            (torch.nn.PReLU(), 'x1 -> x1'),
            (GATConv(hidden_channels, hidden_channels), 'x1, edge_index -> x2'),
            (BatchNorm(hidden_channels), 'x2 -> x2'),
            (torch.nn.PReLU(), 'x2 -> x2'),
            (GATConv(hidden_channels, hidden_channels), 'x2, edge_index -> x3'),
            (BatchNorm(hidden_channels), 'x3 -> x3'),
            (torch.nn.PReLU(), 'x3 -> x3'),
            (lambda x1, x2, x3: [x1, x2, x3], 'x1, x2, x3 -> xs'),
            (JumpingKnowledge('lstm', hidden_channels, num_layers = 3), 'xs -> x'),
            torch.nn.Linear(hidden_channels, classes),
        ]
        )

    def forward(self, x, edge_index):
        return self.jknet(x, edge_index)

# ----------------------- GATs ----------------------------
class GAT_3layer_basic(torch.nn.Module):
    def __init__(self, hidden_channels, input_feat, classes):
        super(GAT_3layer_basic, self).__init__()
        self.GAT1 = GATConv(input_feat, hidden_channels)
        self.GAT2 = GATConv(hidden_channels, hidden_channels)
        self.GAT3 = GATConv(hidden_channels, classes)

    def forward(self, x, edge_index):
        x = self.GAT1(x, edge_index)
        x = x.relu()
        x = self.GAT2(x, edge_index)
        x = x.relu()
        x = self.GAT3(x, edge_index)
        return x

# ----------------------------------------------------------

def train(model, optimizer,
          criterion, data):
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(data.x, data.edge_index)  # Perform a single forward pass.
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss
    
def test(model: torch.nn.Module, data, num_classes = 2, get_auc = False):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    probas = out.softmax(dim=1)[data.test_mask,1].detach().clone().numpy()

    true_Y = data.y[data.test_mask].tolist()

    acc = accuracy_score(true_Y, pred[data.test_mask].tolist())
    if num_classes == 2:
        test_score = f1_score(true_Y, pred[data.test_mask].tolist())
        precision = precision_score(true_Y, pred[data.test_mask].tolist())
        recall = recall_score(true_Y, pred[data.test_mask].tolist())

        # AUROC and AUPRC
        if get_auc:
            auprc = metrics.average_precision_score(true_Y, probas, pos_label = 1)
            auroc = metrics.roc_auc_score(true_Y, probas)

            return test_score, acc, precision, recall, auprc, auroc

    
    return acc, f1_score(true_Y, pred[data.test_mask].tolist())


def val(model: torch.nn.Module, data, num_classes = 2, get_auc = False):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    probas = out.softmax(dim=1)[data.valid_mask,1].detach().clone().numpy()

    true_Y = data.y[data.valid_mask].tolist()

    acc = accuracy_score(true_Y, pred[data.valid_mask].tolist())
    if num_classes == 2:
        test_score = f1_score(true_Y, pred[data.valid_mask].tolist())
        precision = precision_score(true_Y, pred[data.valid_mask].tolist())
        recall = recall_score(true_Y, pred[data.valid_mask].tolist())

        # AUROC and AUPRC
        if get_auc:
            auprc = metrics.average_precision_score(true_Y, probas, pos_label = 1)
            auroc = metrics.roc_auc_score(true_Y, probas)

            return test_score, acc, precision, recall, auprc, auroc

    
    return acc, f1_score(true_Y, pred[data.valid_mask].tolist())

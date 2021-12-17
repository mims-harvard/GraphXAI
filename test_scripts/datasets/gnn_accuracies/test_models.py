import torch
import numpy as np
import scipy.stats as st

from tqdm import trange
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from graphxai.datasets.shape_graph import ShapeGraph
from graphxai.gnn_models.node_classification.testing import GCN_3layer_basic, GIN_3layer_basic
from graphxai.gnn_models.node_classification.testing import GCN_2layer, GIN_2layer
from graphxai.gnn_models.node_classification.testing import GSAGE_3layer, JKNet_3layer

def train_on_split(
        model, 
        optimizer,
        criterion, 
        data,
        split
    ):

    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(data.x, data.edge_index)  # Perform a single forward pass.
    loss = criterion(out[split], data.y[split])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss.item()

def test_on_split(
        model, 
        data, 
        split,
        num_classes = 2
    ):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.

    acc = accuracy_score(data.y[split].tolist(), pred[split].tolist())
    if num_classes == 2:
        test_score = f1_score(data.y[split].tolist(), pred[split].tolist())
        precision = precision_score(data.y[split].tolist(), pred[split].tolist())
        recall = recall_score(data.y[split].tolist(), pred[split].tolist())
        return test_score, acc, precision, recall
    
    return acc

def test_model_on_ShapeGraph(model, epochs_per_run = 200, num_cvs = 50):

    # Cross-validate the model 10 times:
    f1_cv = []
    acc_cv = []
    prec_cv = []
    rec_cv = []

    for i in range(num_cvs):
        # Gen dataset:
        #bah = ShapeGraph(model_layers = 3)
        bah = ShapeGraph(model_layers = 3, num_subgraphs = 75, prob_connection = 0.05)
        data = bah.get_graph()
        # Cross-validation split on dataset nodes:
        kf = KFold(n_splits = 10, shuffle = True)
        nodes = list(range(bah.num_nodes))

        f1_cvi = []
        acc_cvi = []
        prec_cvi = []
        rec_cvi = []

        for train_index, test_index in kf.split(nodes):

            # Set optimizer, loss function:
            modeli = model()
            optimizer = torch.optim.Adam(modeli.parameters(), lr = 0.01)
            criterion = torch.nn.CrossEntropyLoss()

            val_losses = []

            train_idx, val_idx = train_test_split(train_index, test_size = 0.05, shuffle = True)

            for epoch in range(epochs_per_run):
                loss = train_on_split(modeli, optimizer, criterion, data, train_index)

                # Get validation loss:
                val_losses.append(test_on_split(modeli, data, val_idx, num_classes = 2))

                if len(val_losses) > 5:
                    improvement = [int(val_losses[i] >= val_losses[i+1]) for i in range(-5, -1)]

                    if sum(improvement) == 0:
                        break

            print(epoch)
                
            f1, acc, precision, recall = test_on_split(modeli, data, test_index, num_classes = 2)

            f1_cvi.append(f1); acc_cvi.append(acc); prec_cvi.append(precision); rec_cvi.append(recall)

        # Append all means:
        f1_cv.append(np.mean(f1_cvi))
        acc_cv.append(np.mean(acc_cvi))
        prec_cv.append(np.mean(prec_cvi))
        rec_cv.append(np.mean(rec_cvi))

    metrics = ['F1', 'Accuracy', 'Precision', 'Recall']
    results = [f1_cv, acc_cv, prec_cv, rec_cv]
    for i in range(4):
        l = results[i]

        # Get confidence interval:
        ci = st.t.interval(alpha = 0.95, df = len(l) - 1, loc = np.mean(l), scale = st.sem(l))

        print('{} Score: {:.4f} +- {:.4f}'.format(metrics[i], np.mean(l), (ci[1] - ci[0]) / 2))

if __name__ == '__main__':
    model = lambda : GSAGE_3layer(
        hidden_channels=128, 
        input_feat = 10,
        classes = 2)
    print('GSAGE 3 layer, hc = 128')
    # model = lambda : GCN_2layer(
    #     hidden_channels=64, 
    #     input_feat = 10,
    #     classes = 2)

    test_model_on_ShapeGraph(model, epochs_per_run=500)





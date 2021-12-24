import ipdb
import torch
import numpy as np
import scipy.stats as st

from tqdm import trange
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from graphxai.datasets.shape_graph import ShapeGraph
from graphxai.gnn_models.node_classification.testing import GCN_3layer_basic, GIN_3layer_basic
from graphxai.gnn_models.node_classification.testing import GCN_2layer, GIN_2layer

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
    return loss

def test_on_split(
        model,
        criterion,
        data, 
        split,
        num_classes=2
    ):
    model.eval()
    out = model(data.x, data.edge_index)
    loss = criterion(out[split], data.y[split])
    pred = out.argmax(dim=1)  # Use the class with highest probability.

    acc = accuracy_score(data.y[split].tolist(), pred[split].tolist())
    if num_classes == 2:
        test_score = f1_score(data.y[split].tolist(), pred[split].tolist())
        precision = precision_score(data.y[split].tolist(), pred[split].tolist())
        recall = recall_score(data.y[split].tolist(), pred[split].tolist())
        return test_score, acc, precision, recall, loss
    
    return acc


def test_model_on_ShapeGraph(model, epochs_per_run=200, num_cvs=10):

    # Cross-validate the model 10 times:
    f1_cv = []
    acc_cv = []
    prec_cv = []
    rec_cv = []

    for i in trange(num_cvs):
        # Gen dataset:
        bah = ShapeGraph(model_layers=3)
        data = bah.get_graph()
        # ipdb.set_trace()

        # Cross-validation split on dataset nodes:
        kf = KFold(n_splits=10, shuffle=True)
        nodes = list(range(bah.num_nodes))

        f1_cvi = []
        acc_cvi = []
        prec_cvi = []
        rec_cvi = []

        for train_index, test_index in kf.split(nodes):

            # Set optimizer, loss function:
            modeli = model()
            optimizer = torch.optim.Adam(modeli.parameters(), lr=0.01)
            criterion = torch.nn.CrossEntropyLoss()
            tr_loss, te_loss = [], []
            for epoch in range(epochs_per_run):
                train_loss = train_on_split(modeli, optimizer, criterion, data, train_index)
                _, _, _, _, test_loss = test_on_split(modeli, criterion, data, test_index, num_classes=2)
                tr_loss.append(train_loss)
                te_loss.append(test_loss)
                # print(f'Train loss: {train_loss:.5f} || Test loss: {test_loss:.5f}')
            f1, acc, precision, recall, _ = test_on_split(modeli, criterion, data, test_index, num_classes=2)
            # ipdb.set_trace()
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
        ci = st.t.interval(alpha=0.95, df=len(l) - 1, loc=np.mean(l), scale=st.sem(l))

        print('{} Score: {:.4f} +- {:.4f}'.format(metrics[i], np.mean(l), (ci[1] - ci[0]) / 2))


if __name__ == '__main__':
    model = lambda : GIN_3layer_basic(
        hidden_channels=64,
        input_feat=10,
        classes=2)
    # model = lambda : GCN_2layer(
    #     hidden_channels=64, 
    #     input_feat = 10,
    #     classes = 2)

    test_model_on_ShapeGraph(model, epochs_per_run=200)





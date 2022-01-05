import torch
import numpy as np
import scipy.stats as st
import sklearn.metrics as metrics

import warnings
warnings.filterwarnings("ignore")

from tqdm import trange
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from graphxai.datasets import load_ShapeGraph
from graphxai.datasets.shape_graph import ShapeGraph
from graphxai.gnn_models.node_classification.testing import GCN_3layer_basic, GIN_3layer_basic, GCN_4layer_basic, GAT_3layer_basic
from graphxai.gnn_models.node_classification.testing import GCN_2layer, GIN_2layer
from graphxai.gnn_models.node_classification.testing import GSAGE_3layer, JKNet_3layer, JKNet_3layer_lstm

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
        num_classes = 2,
        get_auc = False
    ):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    probas = out.softmax(dim=1)[split,1].numpy()

    true_Y = data.y[split].numpy()

    acc = accuracy_score(true_Y, pred[split].tolist())

    if num_classes == 2:
        test_score = f1_score(true_Y, pred[split].tolist())
        precision = precision_score(true_Y, pred[split].tolist())
        recall = recall_score(true_Y, pred[split].tolist())

        # AUROC and AUPRC
        if get_auc:
            auprc = metrics.average_precision_score(true_Y, probas, pos_label = 1)
            auroc = metrics.roc_auc_score(true_Y, probas)

            # Get AUC's:
            # auprc = metrics.auc(pre, rec)
            # auroc = metrics.auc(fpr, tpr)

            return test_score, acc, precision, recall, auprc, auroc

        return test_score, acc, precision, recall
    
    return acc

def test_model_on_ShapeGraph(model, SG, epochs_per_run = 500, num_cvs = 30):

    # Cross-validate the model 10 times:
    f1_cv = []
    acc_cv = []
    prec_cv = []
    rec_cv = []
    auprc_cv = []
    auroc_cv = []

    #bah = ShapeGraph(model_layers = 3, num_subgraphs = 100, prob_connection = 0.09, subgraph_size=13, max_tries_verification = 15)
    data = SG.get_graph()

    for i in trange(num_cvs):
        # Gen dataset:
        #bah = ShapeGraph(model_layers = 3)

        # Cross-validation split on dataset nodes:
        kf = StratifiedKFold(n_splits = 10, shuffle = True)
        nodes = list(range(SG.num_nodes))

        f1_cvi = []
        acc_cvi = []
        prec_cvi = []
        rec_cvi = []
        auprc_cvi = []
        auroc_cvi = []

        for train_index, test_index in kf.split(nodes, data.y.numpy()):

            # Set optimizer, loss function:
            modeli = model()
            optimizer = torch.optim.Adam(modeli.parameters(), lr = 0.01)
            criterion = torch.nn.CrossEntropyLoss()

            # val_losses = []

            train_idx, val_idx = train_test_split(train_index, test_size = 0.05, shuffle = True, 
                stratify = (data.y[train_index]).numpy())

            for _ in range(epochs_per_run):
                loss = train_on_split(modeli, optimizer, criterion, data, train_index)

            f1, acc, precision, recall, auprc, auroc = test_on_split(modeli, data, test_index, num_classes = 2, get_auc = True)

            f1_cvi.append(f1); acc_cvi.append(acc); prec_cvi.append(precision); rec_cvi.append(recall)
            auprc_cvi.append(auprc); auroc_cvi.append(auroc)

        # Append all means:
        f1_cv.append(np.mean(f1_cvi))
        acc_cv.append(np.mean(acc_cvi))
        prec_cv.append(np.mean(prec_cvi))
        rec_cv.append(np.mean(rec_cvi))
        auprc_cv.append(np.mean(auprc_cvi))
        auroc_cv.append(np.mean(auroc_cvi))

    metrics = ['F1', 'Accuracy', 'Precision', 'Recall', 'AUPRC', 'AUROC']
    results = [f1_cv, acc_cv, prec_cv, rec_cv, auprc_cv, auroc_cv]
    for i in range(len(results)):
        l = results[i]

        # Get confidence interval:
        ci = st.t.interval(alpha = 0.95, df = len(l) - 1, loc = np.mean(l), scale = st.sem(l))

        print('{} Score: {:.4f} +- {:.4f}'.format(metrics[i], np.mean(l), (ci[1] - ci[0]) / 2))

if __name__ == '__main__':
    print('Graph: n=100, p=0.09, sn=13')
    # model = lambda : GSAGE_3layer(
    #     hidden_channels=16, 
    #     input_feat = 10,
    #     classes = 2)
    # print('GSAGE 3 layer, hc = 16, epochs = 150')
    model = lambda : GAT_3layer_basic(
        hidden_channels=16, 
        input_feat = 10,
        classes = 2)
    print('GAT 3 layer, hc = 16')

    SG = load_ShapeGraph(number = 1)
    test_model_on_ShapeGraph(model, SG = SG, epochs_per_run=100, num_cvs = 5)





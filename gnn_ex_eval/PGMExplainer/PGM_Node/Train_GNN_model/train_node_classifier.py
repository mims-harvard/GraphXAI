import models
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import utils
import time
import sklearn.metrics as metrics


def evaluate_node(ypred, labels, train_idx, test_idx):
    _, pred_labels = torch.max(ypred, 2)
    pred_labels = pred_labels.numpy()

    pred_train = np.ravel(pred_labels[:, train_idx])
    pred_test = np.ravel(pred_labels[:, test_idx])
    labels_train = np.ravel(labels[:, train_idx])
    labels_test = np.ravel(labels[:, test_idx])

    result_train = {
        "prec": metrics.precision_score(labels_train, pred_train, average="macro"),
        "recall": metrics.recall_score(labels_train, pred_train, average="macro"),
        "acc": metrics.accuracy_score(labels_train, pred_train),
        "conf_mat": metrics.confusion_matrix(labels_train, pred_train),
    }
    result_test = {
        "prec": metrics.precision_score(labels_test, pred_test, average="macro"),
        "recall": metrics.recall_score(labels_test, pred_test, average="macro"),
        "acc": metrics.accuracy_score(labels_test, pred_test),
        "conf_mat": metrics.confusion_matrix(labels_test, pred_test),
    }
    return result_train, result_test


def train(model,A,X,L,args, normalize_adjacency = False):
    num_nodes = A.shape[0]
    num_train = int(num_nodes * args.train_ratio)
    idx = [i for i in range(num_nodes)]

    np.random.shuffle(idx)
    train_idx = idx[:num_train]
    test_idx = idx[num_train:]
    
    if normalize_adjacency == True:
        A_ = normalize_A(A)
    else:
        A_ = A
        
    # add batch dim
    A_ = np.expand_dims(A_, axis=0)
    X_ = np.expand_dims(X, axis=0)
    L_ = np.expand_dims(L, axis=0)
    
    labels_train = torch.tensor(L_[:, train_idx], dtype=torch.long)
    adj = torch.tensor(A_, dtype=torch.float)
    x = torch.tensor(X_, requires_grad=True, dtype=torch.float)
    scheduler, optimizer = utils.build_optimizer(
        args, model.parameters(), weight_decay=args.weight_decay
    )
    model.train()
    
    ypred = None
    for epoch in range(args.num_epochs):
        begin_time = time.time()
        model.zero_grad()
        ypred, adj_att = model(x, adj)
        ypred_train = ypred[:, train_idx, :]
        loss = model.loss(ypred_train, labels_train)
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()
        elapsed = time.time() - begin_time
        result_train, result_test = evaluate_node(
            ypred.cpu(), L_, train_idx, test_idx
        )
        if epoch % 10 == 0:
            print(
                "epoch: ",
                epoch,
                "; loss: ",
                loss.item(),
                "; train_acc: ",
                result_train["acc"],
                "; test_acc: ",
                result_test["acc"],
                "; train_prec: ",
                result_train["prec"],
                "; test_prec: ",
                result_test["prec"],
                "; epoch time: ",
                "{0:0.2f}".format(elapsed),
            )

        if scheduler is not None:
            scheduler.step()
            
    print(result_train["conf_mat"])
    print(result_test["conf_mat"])
        
    model.eval()
    ypred, _ = model(x, adj)
    
    save_data = {
        "adj": A_,
        "feat": X_,
        "label": L_,
        "pred": ypred.cpu().detach().numpy(),
        "train_idx": train_idx,
    }
    
    utils.save_checkpoint(model, optimizer, args, num_epochs=-1, save_data=save_data)
        
        
        
        
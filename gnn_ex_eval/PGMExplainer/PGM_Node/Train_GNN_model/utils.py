import os
import numpy as np
import torch.optim as optim
import torch
from pathlib import Path

def gen_prefix(args):
    if args.bmname is not None:
        name = args.bmname
    else:
        name = args.dataset
    return name

def create_filename(save_dir, args, isbest=False, num_epochs=-1):
    filename = os.path.join(save_dir, gen_prefix(args))
    if isbest:
        filename = os.path.join(filename, "best")
    elif num_epochs > 0:
        filename = os.path.join(filename, str(num_epochs))
    return filename + ".pth.tar"

def save_checkpoint(model, optimizer, args, num_epochs=-1, isbest=False, save_data=None):
    filename = create_filename(args.ckptdir, args, isbest, num_epochs=num_epochs)
    torch.save(
        {
            "epoch": num_epochs,
            "model_type": args.method,
            "optimizer": optimizer,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "save_data": save_data,
        },
        str(filename),
    )

def load_ckpt(args, isbest=False):
    '''Load a pre-trained pytorch model from checkpoint.
    '''
    print("loading model")
    filename = create_filename(args.ckptdir, args, isbest)
    print(filename)
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        ckpt = torch.load(filename)
    else:
        print("Checkpoint does not exist!")
        print("Checked path -- {}".format(filename))
        print("Make sure you have provided the correct path!")
        print("You may have forgotten to train a model for this dataset.")
        print()
        print("To train one of the paper's models, run the following")
        print(">> python train.py --dataset=DATASET_NAME")
        print()
        raise Exception("File not found.")
    return ckpt

def load_XA(dataname, datadir = "../Generate_XA_Data/XAL"):
    prefix = os.path.join(datadir,dataname)
    filename_A = prefix +"_A.npy"
    filename_X = prefix +"_X.npy"
    A = np.load(filename_A)
    X = np.load(filename_X)
    return A, X

def load_labels(dataname, datadir = "../Generate_XA_Data/XAL"):
    prefix = os.path.join(datadir,dataname)
    filename_L = prefix +"_L.npy"
    L = np.load(filename_L)
    return L

def normalize_A(A):
    sqrt_deg = np.diag(1.0 / np.sqrt(np.sum(A, axis=0, dtype=float).squeeze()))
    A_ = np.matmul(np.matmul(sqrt_deg, A), sqrt_deg)
    return A_

def build_optimizer(args, params, weight_decay=0.0):
    filter_fn = filter(lambda p : p.requires_grad, params)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)
    if args.opt_scheduler == 'none':
        return None, optimizer
    elif args.opt_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    elif args.opt_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.opt_restart)
    return scheduler, optimizer
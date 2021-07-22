import utils
import models
import train_node_classifier
import numpy as np

def task_syn(args):
    A, X = utils.load_XA(args.dataset, datadir = "../Generate_XA_Data/XAL")
    L = utils.load_labels(args.dataset, datadir = "../Generate_XA_Data/XAL")
    num_classes = max(L) + 1
    input_dim = X.shape[1]
    
    model = models.GcnEncoderNode(
            args.input_dim,
            args.hidden_dim,
            args.output_dim,
            num_classes,
            args.num_gc_layers,
            bn=args.bn,
            args=args,
        )                         
    
    train_node_classifier.train(model,A,X,L,args, normalize_adjacency = False)
    
def task_eucore(args):
    A, X = utils.load_XA(args.dataset, datadir = "../Generate_XA_Data/XAL")
    L = utils.load_labels(args.dataset, datadir = "../Generate_XA_Data/XAL")
    num_classes = max(L) + 1
    print("NUMBER OF CLASS IS: " + str(num_classes))
    input_dim = X.shape[1]
    
    model = models.GcnEncoderNode(
            args.input_dim,
            args.hidden_dim,
            args.output_dim,
            num_classes,
            args.num_gc_layers,
            bn=args.bn,
            args=args,
        )                         
    
    train_node_classifier.train(model,A,X,L,args, normalize_adjacency = False)

def task_bitcoinalpha(args):
    A, X = utils.load_XA(args.dataset, datadir = "../Generate_XA_Data/XAL")
    L = utils.load_labels(args.dataset, datadir = "../Generate_XA_Data/XAL")
    num_classes = max(L) + 1
    print("NUMBER OF CLASS IS: " + str(num_classes))
    input_dim = X.shape[1]
    
    print("Input dimension is: ", input_dim)
    
    model = models.GcnEncoderNode(
            input_dim,
            args.hidden_dim,
            args.output_dim,
            num_classes,
            args.num_gc_layers,
            bn=args.bn,
            args=args,
        )                         
    
    train_node_classifier.train(model,A,X,L,args, normalize_adjacency = False)

def task_bitcoinotc(args):
    A, X = utils.load_XA(args.dataset, datadir = "../Generate_XA_Data/XAL")
    L = utils.load_labels(args.dataset, datadir = "../Generate_XA_Data/XAL")
    num_classes = max(L) + 1
    print("NUMBER OF CLASS IS: " + str(num_classes))
    input_dim = X.shape[1]
    
    model = models.GcnEncoderNode(
            input_dim,
            args.hidden_dim,
            args.output_dim,
            num_classes,
            args.num_gc_layers,
            bn=args.bn,
            args=args,
        )                         
    
    train_node_classifier.train(model,A,X,L,args, normalize_adjacency = False)
    
def task_epinions(args):
    A, X = utils.load_XA(args.dataset, datadir = "../Generate_XA_Data/XAL")
    L = utils.load_labels(args.dataset, datadir = "../Generate_XA_Data/XAL")
    num_classes = max(L) + 1
    print("NUMBER OF CLASS IS: " + str(num_classes))
    input_dim = X.shape[1]
    
    model = models.GcnEncoderNode(
            args.input_dim,
            args.hidden_dim,
            args.output_dim,
            num_classes,
            args.num_gc_layers,
            bn=args.bn,
            args=args,
        )                         
    
    train_node_classifier.train(model,A,X,L,args, normalize_adjacency = False)
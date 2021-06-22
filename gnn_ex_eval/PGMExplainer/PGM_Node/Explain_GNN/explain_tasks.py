import utils
import models
import numpy as np
import torch
import pgm_explainer as pe

def task_syn(args):
        
    A, X = utils.load_XA(args.dataset, datadir = "../Generate_XA_Data/XAL")
    L = utils.load_labels(args.dataset, datadir = "../Generate_XA_Data/XAL")
    num_classes = max(L) + 1
    input_dim = X.shape[1]
    ckpt = utils.load_ckpt(args)

    print("input dim: ", input_dim, "; num classes: ", num_classes)
    
    model = models.GcnEncoderNode(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            embedding_dim=args.output_dim,
            label_dim=num_classes,
            num_layers=args.num_gc_layers,
            bn=args.bn,
            args=args,
        )
    
    model.load_state_dict(ckpt["model_state"]) 
    pred = ckpt["save_data"]["pred"]
    
    explainer = pe.Node_Explainer(model, A, X, pred, args.num_gc_layers)
    
    explanations = {}
    if args.explain_node == None:
        if args.dataset == 'syn1': 
            explanations = explainer.explain_range(list(range(300,700)), num_samples = args.num_perturb_samples, top_node = args.top_node)
        elif args.dataset == 'syn2': 
            explanations = explainer.explain_range(list(range(300,700)) + list(range(1000,1400)), num_samples = args.num_perturb_samples, top_node = args.top_node, pred_threshold = 0.1)
        elif args.dataset == 'syn3': 
            explanations = explainer.explain_range(list(range(300,1020)), num_samples = args.num_perturb_samples, top_node = args.top_node,pred_threshold = 0.05) 
        elif args.dataset == 'syn4': 
            explanations = explainer.explain_range(list(range(511,871)), num_samples = args.num_perturb_samples, top_node = args.top_node, pred_threshold = 0.1) 
        elif args.dataset == 'syn5': 
            explanations = explainer.explain_range(list(range(511,1231)), num_samples = args.num_perturb_samples, top_node = args.top_node, pred_threshold = 0.05)     
        elif args.dataset == 'syn6': 
            explanations = explainer.explain_range(list(range(300,700)), num_samples = args.num_perturb_samples, top_node = args.top_node)
    else:
        explanation = explainer.explain(args.explain_node, num_samples = args.num_perturb_samples, top_node = args.top_node)
        print(explanation)
        explanations[args.explain_node] = explanation
    
    
    print(explanations)
    
    savename = utils.gen_filesave(args)
    np.save(savename,explanations)


def bitcoin(args):
        
    A, X = utils.load_XA(args.dataset, datadir = "../Generate_XA_Data/XAL")
    L = utils.load_labels(args.dataset, datadir = "../Generate_XA_Data/XAL")
    num_classes = max(L) + 1
    input_dim = X.shape[1]
    num_nodes = X.shape[0]
    ckpt = utils.load_ckpt(args)

    print("input dim: ", input_dim, "; num classes: ", num_classes)
    
    model = models.GcnEncoderNode(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            embedding_dim=args.output_dim,
            label_dim=num_classes,
            num_layers=args.num_gc_layers,
            bn=args.bn,
            args=args,
        )
    
    model.load_state_dict(ckpt["model_state"]) 
    pred = ckpt["save_data"]["pred"]
    
    explainer = pe.Node_Explainer(model, A, X, pred, 1)
    
    node_to_explain = [i for [i] in np.argwhere(np.sum(A,axis = 0) > 2)]
    
    explanations = explainer.explain_range(node_to_explain, num_samples = args.num_perturb_samples, top_node = args.top_node)
    
    
    print(explanations)
    
    savename = utils.gen_filesave(args)
    np.save(savename,explanations)
    
    

    

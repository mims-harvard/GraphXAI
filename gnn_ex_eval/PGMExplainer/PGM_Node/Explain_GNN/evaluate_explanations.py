import numpy as np
import configs
import utils
import os
import csv
import pandas as pd

def evaluate_bitcoin_explanation(explanations, args):
    # Get predictions
    ckpt = utils.load_ckpt(prog_args)
    pred = ckpt["save_data"]["pred"]
    pred_label = [np.argmax(p) for p in pred[0]]
    
    # Get ground truth
    filename_pos = os.path.join('../Generate_XA_Data/ground_truth_explanation/'+prog_args.dataset,prog_args.dataset+'_pos.csv')
    filename_neg = os.path.join('../Generate_XA_Data/ground_truth_explanation/'+prog_args.dataset,prog_args.dataset+'_neg.csv')
    df_pos = pd.read_csv(filename_pos, header=None, index_col=0, squeeze=True).to_dict()
    df_neg = pd.read_csv(filename_neg, header=None, index_col=0, squeeze=True).to_dict()
    
    # Evaluate
    pred_pos = 0
    true_pos = 0
    for node in explanations:
        gt = []
        if pred_label[node] == 0:
            buff_str = df_neg[node].replace('[','')
            buff_str = buff_str.replace(']','')
            gt = [int(s) for s in buff_str.split(',')]
        else:
            buff_str = df_pos[node].replace('[','')
            buff_str = buff_str.replace(']','')
            gt = [int(s) for s in buff_str.split(',')]
        ex = explanations[node]

        for e in ex:
            pred_pos = pred_pos + 1
            if e in gt:
                true_pos = true_pos + 1

    precision = true_pos/pred_pos
    print("Explainer's precision is ", precision)
    
    savedir = 'result/'
    if args.top_node == None:
        top = "no_top"
    else:
        top = "top_" + str(args.top_node)
    report_file_name = 'report_' + args.dataset + ".txt"
    report_file = os.path.join(savedir, report_file_name)

    with open(report_file, "a") as text_file:
        text_file.write( args.dataset + ", " +  str(args.num_perturb_samples) + " samples, "+ top + " | Precision: " + str(precision) + "\n")
        text_file.write("\n")

def evaluate_syn_explanation(explanations, args):
    gt_positive = 0
    true_positive = 0
    pred_positive = 0
    for node in explanations:
        ground_truth = get_ground_truth(node, args)
        gt_positive = gt_positive + len(ground_truth)
        pred_positive = pred_positive + len(explanations[node])
        for ex_node in explanations[node]:
            if ex_node in ground_truth:
                true_positive = true_positive + 1

    accuracy = true_positive/gt_positive
    precision = true_positive/pred_positive

    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    
    savedir = 'result/'
    if args.top_node == None:
        top = "no_top"
    else:
        top = "top_" + str(args.top_node)
    report_file_name = 'report_' + args.dataset + ".txt"
    report_file = os.path.join(savedir, report_file_name)

    with open(report_file, "a") as text_file:
        text_file.write( prog_args.dataset + ", " +  str(prog_args.num_perturb_samples) + " samples, "+ top + " | Accuracy: " + str(accuracy) + "\n")
        text_file.write( prog_args.dataset + ", " +  str(prog_args.num_perturb_samples) + " samples, "+ top + " | Precision: " + str(precision) + "\n")
        text_file.write("\n")
    

def get_ground_truth(node,args):
    gt = []
    if args.dataset == 'syn1':
        gt =  get_ground_truth_syn1(node) #correct
    elif args.dataset == 'syn2':
        gt =  get_ground_truth_syn1(node) #correct
    elif args.dataset == 'syn3':
        gt =  get_ground_truth_syn3(node) #correct
    elif args.dataset == 'syn4':
        gt =  get_ground_truth_syn4(node) #correct
    elif args.dataset == 'syn5':
        gt =  get_ground_truth_syn5(node) #correct
    elif args.dataset == 'syn6':
        gt =  get_ground_truth_syn1(node) #correct
    return gt

def get_ground_truth_syn1(node):
    base = [0,1,2,3,4]
    ground_truth = []
    offset = node % 5
    ground_truth = [node - offset + val for val in base]   
    return ground_truth

def get_ground_truth_syn3(node):
    base = [0,1,2,3,4,5,6,7,8]
    buff = node - 3
    ground_truth = []
    offset = buff % 9
    ground_truth = [buff - offset + val + 3 for val in base]   
    return ground_truth

def get_ground_truth_syn4(node):
    buff = node - 1
    base = [0,1,2,3,4,5]
    ground_truth = []
    offset = buff % 6
    ground_truth = [buff - offset + val + 1 for val in base]   
    return ground_truth

def get_ground_truth_syn5(node):
    base = [0,1,2,3,4,5,6,7,8]
    buff = node - 7
    ground_truth = []
    offset = buff % 9
    ground_truth = [buff - offset + val + 7 for val in base]   
    return ground_truth

# Get explanations
prog_args = configs.arg_parse()
savename = utils.gen_filesave(prog_args)
explanations = np.load(savename,allow_pickle='TRUE').item()

if prog_args.dataset is not None:
    if prog_args.dataset == "bitcoinalpha":
        evaluate_bitcoin_explanation(explanations, prog_args)
    elif prog_args.dataset == "bitcoinotc":
        evaluate_bitcoin_explanation(explanations, prog_args)
    else:
        evaluate_syn_explanation(explanations, prog_args)

        


























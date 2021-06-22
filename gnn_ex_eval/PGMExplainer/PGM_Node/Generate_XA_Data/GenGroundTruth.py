import os
import networkx as nx
import numpy as np
import configs
import utils
import pandas as pd
import csv

prog_args = configs.arg_parse()

if prog_args.dataset == "bitcoinalpha":
    print("Loading bitcoinalpha dataset")
    datadir = "data"
    dataname = "bitcoinalpha"
    prefix = os.path.join(datadir,dataname)
    filename = prefix + "/bitcoinalpha.csv"
    df = pd.read_csv(filename)
    
    Graphtype = nx.DiGraph()
    G = nx.from_pandas_edgelist(df,  source='SOURCE', target='TARGET', edge_attr='RATING', create_using=Graphtype)
    mapping = {}
    count = 0
    for node in list(G.nodes):
        mapping[node]=count
        count = count + 1
    G=nx.relabel_nodes(G,mapping)

    rating = nx.get_edge_attributes(G,'RATING')
    
    gt_pos = {}
    gt_neg = {}
    for node in list(G.nodes):
        in_edges_list = G.in_edges(node)
        rates = [rating[e] for e in in_edges_list]
        rate_dict = dict(zip(list(in_edges_list), rates))

        top_in_nodes = [node]
        bot_in_nodes = [node]
        for edge in in_edges_list:
            if rating[edge] > 0:
                top_in_nodes.append(edge[0])
            else:
                bot_in_nodes.append(edge[0])


        gt_pos[node] = top_in_nodes
        gt_neg[node] = bot_in_nodes
     
    path_pos = os.path.join('ground_truth_explanation/'+prog_args.dataset,prog_args.dataset+'_pos.csv')
    path_neg = os.path.join('ground_truth_explanation/'+prog_args.dataset,prog_args.dataset+'_neg.csv')
       
    w = csv.writer(open(path_pos, "w"))
    for key, val in gt_pos.items():
        w.writerow([key, val])
    
    w = csv.writer(open(path_neg, "w"))
    for key, val in gt_pos.items():
        w.writerow([key, val])
    
elif prog_args.dataset == "bitcoinotc":
    print("Loading bitcoinotc dataset")
    datadir = "data"
    dataname = "bitcoinotc"
    prefix = os.path.join(datadir,dataname)
    filename = prefix + "/bitcoinotc.csv"
    df = pd.read_csv(filename)
    
    Graphtype = nx.DiGraph()
    G = nx.from_pandas_edgelist(df,  source='SOURCE', target='TARGET', edge_attr='RATING', create_using=Graphtype)
    mapping = {}
    count = 0
    for node in list(G.nodes):
        mapping[node]=count
        count = count + 1
    G=nx.relabel_nodes(G,mapping)

    rating = nx.get_edge_attributes(G,'RATING')
    
    gt_pos = {}
    gt_neg = {}
    for node in list(G.nodes):
        in_edges_list = G.in_edges(node)
        rates = [rating[e] for e in in_edges_list]
        rate_dict = dict(zip(list(in_edges_list), rates))

        top_in_nodes = [node]
        bot_in_nodes = [node]
        for edge in in_edges_list:
            if rating[edge] > 0:
                top_in_nodes.append(edge[0])
            else:
                bot_in_nodes.append(edge[0])


        gt_pos[node] = top_in_nodes
        gt_neg[node] = bot_in_nodes
     
    path_pos = os.path.join('ground_truth_explanation/'+prog_args.dataset,prog_args.dataset+'_pos.csv')
    path_neg = os.path.join('ground_truth_explanation/'+prog_args.dataset,prog_args.dataset+'_neg.csv')
       
    w = csv.writer(open(path_pos, "w"))
    for key, val in gt_pos.items():
        w.writerow([key, val])
    
    w = csv.writer(open(path_neg, "w"))
    for key, val in gt_pos.items():
        w.writerow([key, val]) 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
""" utils.py
    Utilities for reading data.
"""

import os

import numpy as np
import pandas as pd
import scipy as sc
import pandas as pd

import numpy as np
import networkx as nx

import featgen

def save_XAL(G,labels,prog_args):
    # Save the feature matrix, the adjacency matrix and the labels list
    A = nx.adjacency_matrix(G).todense()
    X = np.asarray([G.nodes[node]['feat'] for node in list(G.nodes)])
    pathA = os.path.join('XAL',prog_args.dataset+'_A')
    pathX = os.path.join('XAL',prog_args.dataset+'_X')
    pathL = os.path.join('XAL',prog_args.dataset+'_L')
    np.save(pathA,A)
    np.save(pathX,X)
    np.save(pathL,labels)

def read_bitcoinotc(datadir = "data",dataname ="bitcoinotc",feature_generator=None):
    prefix = os.path.join(datadir,dataname)
    filename = prefix + "/bitcoinotc.csv"
    
    df = pd.read_csv(filename)
    
    Graphtype = nx.DiGraph()
    G = nx.from_pandas_edgelist(df,  source='SOURCE', target='TARGET', edge_attr='RATING', create_using=Graphtype)
    
    mapping = {}
    count = 0
    for node in list(G.nodes):
        count = count + 1
        mapping[node]=count
    G=nx.relabel_nodes(G,mapping)
    
    rating = nx.get_edge_attributes(G,'RATING')
    max_rating = rating[max(rating, key=rating.get)]
    degree_sequence_in = [d for n, d in G.in_degree()]
    dmax_in = max(degree_sequence_in)
    degree_sequence_out = [d for n, d in G.out_degree()]
    dmax_out = max(degree_sequence_out)
    
    label_mapping = {}
    rate_mapping = {}
    decision_threshold = 0.3
    number_of_in_nodes_threshold = 3
    
    for node in list(G.nodes):
        in_edges_list = G.in_edges(node)
        if len(in_edges_list)  < number_of_in_nodes_threshold:
            total_rate = 0
            label = 0
            rate_mapping[node] = 0
            label_mapping[node] = label
        else:
            total_rate = 0
            for (source,_) in in_edges_list:
                total_rate = total_rate + G.get_edge_data(source,node)['RATING']/ np.abs(G.get_edge_data(source,node)['RATING'])
            average_rate = total_rate/len(in_edges_list)

            label = 0
            if average_rate < decision_threshold:
                label = 0
            else:
                label = 1
                
            rate_mapping[node] = average_rate
            label_mapping[node] = label
            
    roles = []
    count = 0
    count1 = 0
    for node, l in label_mapping.items():
        count = count + 1
        if l == 1:
            count1 = count1 + 1
        roles.append(l)
    print("Total node: ", count)
    print("Positive node: ", count1)
    
    if feature_generator is None:
        
        feat_dict = {}
        feature_length = 8
        for node in list(G.nodes):
            out_edges_list = G.out_edges(node)
            
            if len(out_edges_list) == 0:     
                features = np.ones(feature_length, dtype=float)/1000
                feat_dict[node] = {'feat': features}
            else:
                features = np.zeros(feature_length, dtype=float)
                w_pos = 0
                w_neg = 0
                for (_,target) in out_edges_list:
                    w = G.get_edge_data(node,target)['RATING']
                    if w >= 0:
                        w_pos = w_pos + w
                    else:
                        w_neg = w_neg - w
                
                abstotal = (w_pos + w_neg)
                average = (w_pos - w_neg)/len(out_edges_list)/max_rating
                
                features[0] = w_pos/max_rating/len(out_edges_list) # average positive vote
                features[1] = w_neg/max_rating/len(out_edges_list) # average negative vote
                features[2] = w_pos/abstotal 
                features[3] = average
                features[4] = features[0]*G.in_degree(node)/dmax_in
                features[5] = features[1]*G.in_degree(node)/dmax_in
                features[6] = features[0]*G.out_degree(node)/dmax_out
                features[7] = features[1]*G.out_degree(node)/dmax_out
                         
                features = features/1.01 + 0.001
                
                feat_dict[node] = {'feat': features}  
        print("Good nodes ratio: ", count1/count)

        nx.set_node_attributes(G, feat_dict)
    else:
        feature_generator.gen_node_features(G)
        
    name = "bitcoinotc"
    G = G.to_undirected()
    
    return G, roles, name

    
def read_bitcoinalpha(datadir = "data",dataname ="bitcoinalpha",feature_generator=None):
    prefix = os.path.join(datadir,dataname)
    filename = prefix + "/bitcoinalpha.csv"
    
    df = pd.read_csv(filename)
    
    Graphtype = nx.DiGraph()
    G = nx.from_pandas_edgelist(df,  source='SOURCE', target='TARGET', edge_attr='RATING', create_using=Graphtype)
    
    mapping = {}
    count = 0
    for node in list(G.nodes):
        count = count + 1
        mapping[node]=count
    G=nx.relabel_nodes(G,mapping)
    
    rating = nx.get_edge_attributes(G,'RATING')
    max_rating = rating[max(rating, key=rating.get)]
    degree_sequence_in = [d for n, d in G.in_degree()]
    dmax_in = max(degree_sequence_in)
    degree_sequence_out = [d for n, d in G.out_degree()]
    dmax_out = max(degree_sequence_out)
    
    label_mapping = {}
    rate_mapping = {}
    decision_threshold = 0.3
    number_of_in_nodes_threshold = 3
    
    for node in list(G.nodes):
        in_edges_list = G.in_edges(node)
        if len(in_edges_list)  < number_of_in_nodes_threshold:
            total_rate = 0
            label = 0
            rate_mapping[node] = 0
            label_mapping[node] = label
        else:
            total_rate = 0
            for (source,_) in in_edges_list:
                total_rate = total_rate + G.get_edge_data(source,node)['RATING']/ np.abs(G.get_edge_data(source,node)['RATING'])
            average_rate = total_rate/len(in_edges_list)

            label = 0
            if average_rate < decision_threshold:
                label = 0
            else:
                label = 1
                
            rate_mapping[node] = average_rate
            label_mapping[node] = label
            
    roles = []
    count = 0
    count1 = 0
    for node, l in label_mapping.items():
        count = count + 1
        if l == 1:
            count1 = count1 + 1
        roles.append(l)
    print("Total node: ", count)
    print("Positive node: ", count1)
    
    if feature_generator is None:
        
        feat_dict = {}
        feature_length = 8
        for node in list(G.nodes):
            out_edges_list = G.out_edges(node)
            
            if len(out_edges_list) == 0:     
                features = np.ones(feature_length, dtype=float)/1000
                feat_dict[node] = {'feat': features}
            else:
                features = np.zeros(feature_length, dtype=float)
                w_pos = 0
                w_neg = 0
                for (_,target) in out_edges_list:
                    w = G.get_edge_data(node,target)['RATING']
                    if w >= 0:
                        w_pos = w_pos + w
                    else:
                        w_neg = w_neg - w
                
                abstotal = (w_pos + w_neg)
                average = (w_pos - w_neg)/len(out_edges_list)/max_rating
                
                features[0] = w_pos/max_rating/len(out_edges_list) # average positive vote
                features[1] = w_neg/max_rating/len(out_edges_list) # average negative vote
                features[2] = w_pos/abstotal 
                features[3] = average
                features[4] = features[0]*G.in_degree(node)/dmax_in
                features[5] = features[1]*G.in_degree(node)/dmax_in
                features[6] = features[0]*G.out_degree(node)/dmax_out
                features[7] = features[1]*G.out_degree(node)/dmax_out
                         
                features = features/1.01 + 0.001
                
                feat_dict[node] = {'feat': features}  
        print("Good nodes ratio: ", count1/count)

        nx.set_node_attributes(G, feat_dict)
    else:
        feature_generator.gen_node_features(G)
        
    name = "bitcoinalpha"
    G = G.to_undirected()
    
    return G, roles, name

def read_epinions(datadir = "data",dataname ="epinions",feature_generator=None):
    prefix = os.path.join(datadir,dataname)
    filename = prefix + "/epinions.csv"
    
    df = pd.read_csv(filename)
    
    Graphtype = nx.DiGraph()
    G = nx.from_pandas_edgelist(df,  source='SOURCE', target='TARGET', edge_attr='RATING', create_using=Graphtype)
    
    mapping = {}
    count = 0
    for node in list(G.nodes):
        count = count + 1
        mapping[node]=count
    G=nx.relabel_nodes(G,mapping)
    
    rating = nx.get_edge_attributes(G,'RATING')
    
    rate_mapping = {}
    label_mapping = {}
    rank = [-0.1,0.25]
    for node in list(G.nodes):
        in_edges_list = G.in_edges(node)
        if len(in_edges_list) == 0:
            total_rate = 0
            label = 0
            if total_rate <= rank[0]:
                label = 0
            else:
                for r in range(len(rank)-1):
                    if total_rate > rank[r]:    
                        if total_rate <= rank[r+1]:
                            label = r+1
                if total_rate > rank[len(rank)-1]:
                    label = len(rank)
            rate_mapping[node] = total_rate
            rate_mapping[node]=0
            label_mapping[node] = label
        else:
            total_rate = 0
            for (source,_) in in_edges_list:
                total_rate = total_rate + G.get_edge_data(source,node)['RATING']
            total_rate = total_rate/len(in_edges_list)

            label = 0
            if total_rate <= rank[0]:
                label = 0
            else:
                for r in range(len(rank)-1):
                    if total_rate > rank[r]:    
                        if total_rate <= rank[r+1]:
                            label = r+1
                if total_rate > rank[len(rank)-1]:
                    label = len(rank)
            rate_mapping[node] = total_rate
            label_mapping[node] = label
            
    roles = [] 
    for node, l in label_mapping.items():
        roles.append(l)
    
    if feature_generator is None:
        feature_generator = featgen.ConstFeatureGen(1)
    feature_generator.gen_node_features(G)
    
    name = "epinions"
    G = G.to_undirected()    
    return G, roles, name

def read_eucore(datadir = "data",dataname ="eucore",feature_generator=None):
    prefix = os.path.join(datadir,dataname)
    filename_edges = prefix + "/email-Eu-core.txt"

    A = []
    try:
        with open(filename_edges) as f:
            for line in f:
                line = line.strip("\n").split(" ")
                e0, e1 = (int(line[0].strip(" ")), int(line[1].strip(" ")))
                A.append((e0,e1))
    except IOError:
        print("No Eucore edge data.")

    roles = []
    filename_labels = prefix + "/email-Eu-core-labels.txt"
    try:
        with open(filename_labels) as f:
            for line in f:
                line = line.strip("\n").split(" ")
                node, role = (int(line[0].strip(" ")), int(line[1].strip(" ")))
                roles.append(role)
    except IOError:
        print("No Eucore label data.")

    G = nx.from_edgelist(A)
    if feature_generator is None:
        feature_generator = featgen.ConstFeatureGen(1)
    feature_generator.gen_node_features(G)

    name = "eucore"
    return G, roles, name

def read_amazon(datadir = "data",dataname ="amazon",feature_generator = None):
    prefix = os.path.join(datadir,dataname)
    filename_edges = prefix + "/com-amazon.ungraph.txt"

    A = []
    try:
        with open(filename_edges) as f:
            for line in f:
                line = line.strip("\n").split("\t")
                e0, e1 = (int(line[0].strip(" ")), int(line[1].strip(" ")))
                A.append((e0,e1))
    except IOError:
        print("No Amazon edge data.")

    G = nx.from_edgelist(A)
    
    G_map = {n:i for i,n in enumerate(G.nodes)}
    G = nx.relabel_nodes(G, G_map)
    num_nodes = G.number_of_nodes()
    
    X = []
    filename_labels = prefix + "/com-amazon.top5000.cmty.txt"
    try:
        with open(filename_labels) as f:
            for line in f:
                line = line.strip("\n").split("\t")
                line_int = [G_map[int(i)] for i in line]
                feat_vector = np.zeros(num_nodes)
                feat_vector[line_int] = 1
                X.append(feat_vector)
    except IOError:
        print("No Eucore label data.")
    
    X = np.transpose(np.asarray(X))
    feat_dict = {i:{'feat': X[i]} for i in G.nodes()}          
    nx.set_node_attributes(G, feat_dict)

    name = "amazon"
    return G, A, X, name

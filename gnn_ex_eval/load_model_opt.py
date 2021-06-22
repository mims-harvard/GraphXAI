#%%
import dgl
import ipdb
import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import *
from models import *
import torch_geometric.transforms as T
from sklearn.metrics import f1_score, roc_auc_score
from torch_geometric.utils import dropout_adj, convert
from torch_geometric.nn import GCNConv, SAGEConv, GINConv
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator


class ModelDataLoader():
	def __init__(self, args):
		self.model = args.model
		self.nhid = args.hidden
		self.dropout = args.dropout
		self.lr = args.lr
		self.weight_decay = args.weight_decay
		self.num_heads = args.num_heads
		self.num_layers = args.num_layers
		self.num_out_heads = args.num_out_heads

		self.dataset = args.dataset

	def load_model_optim(self, nfeat, num_class):
		if self.model == 'gcn':
			model = GCN(nfeat=nfeat,
						nhid=self.nhid,
						nclass=num_class,
						dropout=self.dropout)
		elif self.model == 'gat':
			heads =  ([self.num_heads] * self.num_layers) + [self.num_out_heads]
			model = GAT(num_layers=self.num_layers, num_classes=num_class, in_dim=nfeat,
						num_hidden=self.nhid, heads=heads, feat_drop=0.0,
						attn_drop=0.0, negative_slope=0.2, residual=False)
		elif self.model == 'sage':
			model = SAGE(nfeat=nfeat,
						nhid=self.nhid,
						nclass=num_class,
						dropout=self.dropout)
		elif self.model == 'gin':
			model = GIN(nfeat=nfeat,
						nhid=self.nhid,
						nclass=num_class,
						dropout=self.dropout)
		elif self.model == 'jk':
			model = JK(nfeat=nfeat,
						nhid=self.nhid,
						nclass=num_class,
						dropout=self.dropout)
		elif self.model == 'infomax':
			enc_dgi = Encoder_DGI(nfeat=nfeat, nhid=self.nhid)
			enc_cls = Encoder_CLS(nhid=self.nhid, nclass=num_class)
			model = GraphInfoMax(enc_dgi=enc_dgi, enc_cls=enc_cls)

		optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

		return model, optimizer

	def load_dataset(self):

		# Load credit_scoring dataset
		if self.dataset == 'credit':
			sens_attr = "Age"  # column number after feature process is 1
			sens_idx = 1
			predict_attr = 'NoDefaultNextMonth'
			label_number = 6000
			path_credit = "./dataset/credit"
			adj, features, labels, idx_train, idx_val, idx_test, sens = load_credit(self.dataset, sens_attr,
																					predict_attr, path=path_credit,
																					label_number=label_number
																					)
			norm_features = feature_norm(features)
			set_categorical_mask = []
			for i in range(features.shape[1]):
				if features[:, i].unique().shape[0] <= 10:
					set_categorical_mask.append([i, 1, features[:, i].unique().shape[0]])
					norm_features[:, i] = features[:, i]
				else:
					set_categorical_mask.append([i, 0, 0])
			features = norm_features

		# Load german dataset
		elif self.dataset == 'german':
			sens_attr = "Gender"  # column number after feature process is 0
			sens_idx = 0
			predict_attr = "GoodCustomer"
			label_number = 100
			path_german = "./dataset/german"
			adj, features, labels, idx_train, idx_val, idx_test, sens = load_german(self.dataset, sens_attr,
																					predict_attr, path=path_german,
																					label_number=label_number,
																					)
			norm_features = feature_norm(features)
			set_categorical_mask = []
			for i in range(features.shape[1]):
				if features[:, i].unique().shape[0] <= 10:
					set_categorical_mask.append([i, 1, features[:, i].unique().shape[0]])
					norm_features[:, i] = features[:, i]
				else:
					set_categorical_mask.append([i, 0, 0])
			features = norm_features

		# Load bail dataset
		elif self.dataset == 'bail':
			sens_attr = "WHITE"  # column number after feature process is 0
			sens_idx = 0
			predict_attr = "RECID"
			label_number = 100
			path_bail = "./dataset/bail"
			adj, features, labels, idx_train, idx_val, idx_test, sens = load_bail(self.dataset, sens_attr,
											predict_attr, path=path_bail,
																					label_number=label_number,
																					)

			norm_features = feature_norm(features)
			set_categorical_mask = []
			for i in range(features.shape[1]):
				if features[:, i].unique().shape[0] <= 10:  # < 5
					set_categorical_mask.append([i, 1, features[:, i].unique().shape[0]])
					norm_features[:, i] = features[:, i]
				else:
					set_categorical_mask.append([i, 0, 0])
			features = norm_features

		# Load ogbn-arxiv dataset
		elif self.dataset == 'arxiv':
			dataset = PygNodePropPredDataset(name='ogbn-arxiv',
											 transform=T.ToSparseTensor())

			data = dataset[0]
			data.adj_t = data.adj_t.to_symmetric()
			split_idx = dataset.get_idx_split()
			idx_train = split_idx['train']
			idx_val = split_idx['valid']
			idx_test = split_idx['test']
			labels = data.y.squeeze(1)
			sens = None
			sens_idx = None
			set_categorical_mask = [0 for _ in range(data.x.shape[1])]
			adj = data.adj_t
			features = data.x

		else:
			print('Invalid dataset name!!')
			exit(0)

		return adj, features, labels, idx_train, idx_val, idx_test, sens, sens_idx, torch.from_numpy(np.array(set_categorical_mask))

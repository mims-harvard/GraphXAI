import math
import time
import os
import ipdb

import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import tensorboardX.utils

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score, recall_score, precision_score, roc_auc_score, precision_recall_curve
from sklearn.cluster import DBSCAN


use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor


class VanillaGrad:
	def __init__(self, model, label, x, edge_index, mapping, node_idx, subset, device):
		self.model = model
		self.label = label
		self.x = x
		self.edge_index = edge_index
		self.mapping = mapping
		self.node_idx = node_idx
		self.subset = subset
		self.device = device

	def __set_masks__(self, x, edge_index, init="normal"):
		pass

	def explain(self, x):
		"""Explain a single node prediction
		"""
		self.model.eval()
		output = self.model(x.to(self.device), self.edge_index.to(self.device))

		# NLL_loss
		loss = F.nll_loss(output[self.mapping], self.label[self.mapping].to(self.device))
		loss.backward()

		return x.grad[torch.where(self.subset==self.node_idx)[0].item(), :]

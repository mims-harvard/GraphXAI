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

from graphlime import GraphLIME

import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score, recall_score, precision_score, roc_auc_score, precision_recall_curve
from sklearn.cluster import DBSCAN


use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor


class GLIME:
	def __init__(self, model, x, node_idx, edge_index, hop, rho, device):
		self.model = model
		self.x = x
		self.node_idx = node_idx
		self.edge_index = edge_index
		self.hop = hop
		self.rho = rho
		self.device = device

	def explain(self, x):
		"""Explain a single node prediction
		"""
		self.model = self.model.to('cpu')
		explainer = GraphLIME(self.model, hop=self.hop, rho=self.rho)
		explanation, tr_K, tr_L = explainer.explain_node(self.node_idx, x, self.edge_index)
		return torch.from_numpy(explanation), tr_K, tr_L  # .to(self.device)

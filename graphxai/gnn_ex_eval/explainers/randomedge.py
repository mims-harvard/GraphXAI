import ipdb
import torch


class RandomEdges:
	def __init__(self):
		pass

	def __set_masks__(self):
		pass

	def explain(self, x, edge_index):
		"""Explain a single node prediction using edges
		"""
		return torch.randn(edge_index[0, :].shape)

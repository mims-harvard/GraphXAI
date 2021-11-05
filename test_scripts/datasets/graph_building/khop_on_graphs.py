import torch
import networkx as nx

from graphxai.utils import khop_subgraph_nx
from graphxai.datasets import BAShapes

class Hyperparameters:
    num_hops = 2
    n = 30 # Invalid for bound
    m = 1 # Invalid for bound
    num_shapes = 5 # Invalid
    shape_insert_strategy = 'bound_12'
    shape_upper_bound = 1
    shape = 'house'
    labeling_method = 'edge'
    feature_method = 'gaussian_lv'

hyp = Hyperparameters
args = {key:value for key, value in hyp.__dict__.items() if not key.startswith('__') and not callable(value)}
bah = BAShapes(**args)

bah.visualize()
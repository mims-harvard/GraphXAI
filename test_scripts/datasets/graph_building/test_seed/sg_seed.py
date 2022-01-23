import torch
import matplotlib.pyplot as plt

from graphxai.datasets import ShapeGraph

arg_dict = {
    'num_subgraphs': 5,
    'prob_connection': 0.25,
    'subgraph_size': 8
}

fig, (ax1, ax2) = plt.subplots(1, 2)

sg1 = ShapeGraph(**arg_dict, model_layers = 3, seed = 8001)
sg1.visualize(shape_label = True, ax = ax1)

sg2 = ShapeGraph(**arg_dict, model_layers = 3, seed = 8001)
sg2.visualize(shape_label = True, ax = ax2, show = True)

print('x diff:', (sg1.get_graph().x - sg2.get_graph().x))
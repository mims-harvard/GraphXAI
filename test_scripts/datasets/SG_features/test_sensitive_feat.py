import torch

from graphxai.datasets import ShapeGraph
from sklearn.metrics import confusion_matrix

SG = ShapeGraph(
    model_layers = 3,
    num_subgraphs = 100,
    prob_connection = 0.08,
    subgraph_size = 10,
    class_sep = 30,
    n_informative = 2,
    verify = False,
    make_explanations = False,
    homphily_coef = -1.0,
    attribute_sensitive_feature = True,
    sens_attribution_noise = 0.8
)

data = SG.get_graph(use_fixed_split=True)

sensitive = data.x[:, SG.sensitive_feature]

print('sensitive vs. y')
print(confusion_matrix(sensitive.int().tolist(), data.y.int().tolist()))
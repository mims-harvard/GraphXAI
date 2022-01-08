import os
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from graphxai.datasets import load_ShapeGraph, ShapeGraph

root_data = os.path.join('/Users/owenqueen/Desktop/data', 'ShapeGraph')

#SG = load_ShapeGraph(number=1, root = root_data)
SG = ShapeGraph(
    model_layers = 3,
    num_subgraphs = 100,
    prob_connection = 0.08,
    subgraph_size = 10,
    class_sep = 1,
    n_informative = 4,
    flip_y = 0,
    verify = False
)

data = SG.get_graph()

print('X size', data.x.shape)

X = data.x.numpy()
Y = data.y.numpy()

print(f'Num (0): {np.sum(Y == 0)}')
print(f'Num (1): {np.sum(Y == 1)}')

parameters = {
    'C': list(np.arange(0.25, 1.5, step=0.25)),
}

#lr = LogisticRegression()
svm = SVC()
clf = GridSearchCV(svm, parameters, scoring='roc_auc', verbose = 1)

clf.fit(X, Y)

print('Best ROC-AUC', clf.best_score_)
print('Best params', clf.best_params_)
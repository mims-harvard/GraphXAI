import numpy as np

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

X, Y = make_classification(
    n_samples = 1500,
    n_features = 10,
    class_sep = 50,
    scale = 1,
    n_informative = 4,
    n_redundant = 0,
    flip_y = 0,
    weights = [0.33, 0.67],
)

parameters = {
    'C': list(np.arange(0.25, 1.5, step=0.25)),
}

lr = LogisticRegression(C = 0.25)
clf = GridSearchCV(lr, parameters, scoring='roc_auc', verbose = 1)

clf.fit(X, Y)

print('Best F1', clf.best_score_)
print('Best params', clf.best_params_)
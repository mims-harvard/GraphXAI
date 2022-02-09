import matplotlib.pyplot as plt
#import seaborn as sns; sns.set()
import numpy as np

# Generate some data
from sklearn.datasets import make_blobs
X, y_true = make_blobs(n_samples=200, centers=2,
                       cluster_std=2, random_state=0)
X = X[:, ::-1] # flip axes for better plotting

c = {0: 'grey', 1: 'red'}
colors = [c[yi] for yi in y_true]

# Plot the data with K Means Labels
plt.scatter(X[:, 0], X[:, 1], c=colors, s=40, cmap='viridis', alpha = 0.5)
plt.show()
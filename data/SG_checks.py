# Get statistics for a given file (pickle) of SG instance
import os, argparse
import torch
import numpy as np
from graphxai.datasets import load_ShapeGraph
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from graphxai.gnn_models.node_classification.testing import GIN_3layer_basic, test

# Replace with your root
# my_root = os.path.join('/home/owq978/GraphXAI/',
#     'data', 'ShapeGraph', #'unzipped',
# )

my_root = os.path.join('/Users/owenqueen/Desktop/HMS_research/graphxai_project/GraphXAI/data',
    'ShapeGraph')

attr_list = [
    'variant',
    'base_graph',
    'verify',
    'max_tries_verification',
    'n_informative',
    'n_features',
    'n_clusters_per_class',
    'add_sensitive_feature',
    'attribute_sensitive_feature',
    'sens_attribution_noise'
]

core_attr = [
    'num_nodes',
    'sensitive_feature',
    'homophily_coef',
    'seed',
    'num_subgraphs',
    'prob_connection',
    'subgraph_size',
    'class_sep',
]

def iter_attr_list(obj, attrs):
    all_attrs = dict()
    for a in attrs:
        try:
            aget = getattr(obj, a)
        except:
            aget = 'None'
        
        all_attrs[a] = aget

    return all_attrs

def get_stats(fname, gnn = None):
    
    SG = load_ShapeGraph(fname, root = my_root)

    data = SG.get_graph(use_fixed_split = True)

    print('ShapeGraph stats: --------------------------------')

    # Get class imbalances:
    num0 = (data.y == 0).nonzero(as_tuple=True)[0].shape[0]
    num1 = (data.y == 1).nonzero(as_tuple=True)[0].shape[0]

    print('Class 0 (total): {}'.format(num0))
    print('Class 1 (total): {}'.format(num1))

    # Get size of test and train splits:
    num_test = data.y[data.test_mask].shape[0]
    num_train = data.y[data.train_mask].shape[0]

    print('')
    print('Num test:  {}'.format(num_test))
    print('Num train: {}'.format(num_train))
    print('')

    # Get num edges (not stored natively):
    num_edges = data.edge_index.shape[1]
    print('{} {}'.format('num_edges'.ljust(25), num_edges))

    deglist = [SG.G.degree[i] for i in SG.G.nodes]
    print('{} {:.3f} +- {:.4f}'.format('avg node deg'.ljust(25), \
            sum(deglist) / SG.G.number_of_nodes(), \
            np.std(deglist) / np.sqrt(len(deglist))))

    # Get num features (actual)
    num_feat = data.x.shape[1]
    print('{} {}'.format('actual num_feat'.ljust(25), num_feat))

    core = iter_attr_list(SG, core_attr)
    for k, v in core.items():
        print('{} {}'.format(k.ljust(25), v))

    print('')
    other = iter_attr_list(SG, attr_list)
    for k, v in other.items():
        print('{} {}'.format(k.ljust(25), v))

    # Train logistic regression and GNN:
    parameters = {
        'C': list(np.arange(0.25, 1.5, step=0.25)),
    }

    lr = LogisticRegression()
    clf = GridSearchCV(lr, parameters, scoring='roc_auc')
    X = data.x.detach().clone().numpy()
    Y = data.y.numpy()
    clf.fit(X, Y)

    print('')
    print('LogisticRegression Best AUROC', clf.best_score_)
    print('LogisticRegression Best params', clf.best_params_)

    # Plot degtee distribution
    degrees = sorted([d for n, d in SG.G.degree()])

    variant_code = 'PA'

    plt.hist(degrees, color = 'green')
    plt.title('Degree Distribution - {}'.format(variant_code))
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.show()

    # Get GNN model, if provided:
    if gnn is not None:
        model = GIN_3layer_basic(16, input_feat = 11, classes = 2)
        model.load_state_dict(torch.load(gnn))
        f1, acc, pre, rec, auprc, auroc = test(model, data, num_classes = 2, get_auc=True)

        print('GNN (GIN, 3 layer) AUROC = {:.4f}, F1 = {:.4f}'.format(auroc, f1))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', type = str, required = True, help = 'Base name of ShapeGraph to show')
    parser.add_argument('--model_path', type=str, default = None, help = 'path to model trained for this task')
    
    args = parser.parse_args()

    get_stats(args.fname, gnn = args.model_path)

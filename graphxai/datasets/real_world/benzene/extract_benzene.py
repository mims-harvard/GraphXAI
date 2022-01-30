import os
from sympy import E
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch_geometric.data import Data

from graphxai.utils import Explanation, edge_mask_from_node_mask, aggregate_explanations

benzene_data_dir = os.path.join(os.path.dirname(__file__), 'benzene_data')

def load_graphs(dir_path = benzene_data_dir):
    '''
    TODO: replace path with Harvard Dataverse loading

    Loads all graphs for benzene in a way that they can be directly fed to 
        dataset class.
    '''
    
    att = np.load(os.path.join(dir_path, 'true_raw_attribution_datadicts.npz'),
            allow_pickle = True)
    X = np.load(os.path.join(dir_path, 'x_true.npz'), allow_pickle = True)
    y = np.load(os.path.join(dir_path, 'y_true.npz'), allow_pickle = True)
    ylist = [y['y'][i][0] for i in range(y['y'].shape[0])]

    att = att['datadict_list']
    X = X['datadict_list'][0]

    df = pd.read_csv(os.path.join(dir_path, 'benzene_smiles.csv'))

    # Unique zinc identifiers:
    zinc_ids = df['mol_id'].tolist()

    all_graphs = []
    explanations = []

    print('Len X', len(X))

    # n_imp_sizes0 = []
    # n_imp_sizes1 = []

    for i in range(len(X)):
        x = torch.from_numpy(X[i]['nodes'])
        edge_attr = torch.from_numpy(X[i]['edges'])
        #y = X[i]['globals'][0]
        y = torch.tensor([ylist[i]], dtype = torch.long)

        # Get edge_index:
        e1 = torch.from_numpy(X[i]['receivers']).long()
        e2 = torch.from_numpy(X[i]['senders']).long()

        edge_index = torch.stack([e1, e2])

        data_i = Data(
            x = x,
            y = y,
            edge_attr = edge_attr,
            edge_index = edge_index
        )

        all_graphs.append(data_i) # Add to larger list

        # Get ground-truth explanation:
        node_imp = torch.from_numpy(att[i][0]['nodes']).float()

        # Error-check:
        assert att[i][0]['n_edge'] == X[i]['n_edge'], 'Num: {}, Edges different sizes'.format(i)
        assert node_imp.shape[0] == x.shape[0], 'Num: {}, Shapes: {} vs. {}'.format(i, node_imp.shape[0], x.shape[0]) \
            + '\nExp: {} \nReal:{}'.format(att[i][0], X[i])

        i_exps = []

        for j in range(node_imp.shape[1]):

        # n_imp_sizes0.append(node_imp.shape[0])
        # n_imp_sizes1.append(node_imp.shape[1])

            exp = Explanation(
                feature_imp = None, # No feature importance - everything is one-hot encoded
                node_imp = node_imp,
                edge_imp = edge_mask_from_node_mask(node_imp.bool(), edge_index = edge_index),
            )
            
            exp.set_whole_graph(data_i)
            i_exps.append(exp)
            
        explanations.append(i_exps)

    # plt.hist(n_imp_sizes0, bins = 20)
    # plt.title('0')
    # plt.show()

    # plt.hist(n_imp_sizes1, bins = 20)
    # plt.title('1')
    # plt.show()

    return all_graphs, explanations, zinc_ids

if __name__ == '__main__':
    # Test if it runs:
    ag, exp, zinc = load_graphs('benzene_data')

    print(len(ag))
    print(len(exp))
    print(len(zinc))

    # Choose one that has at least two rings:
    i = 0
    while len(exp[i]) <= 1:
        i += 1 

    aggregate_explanations(exp[i], node_level = False).graph_draw(show = True)
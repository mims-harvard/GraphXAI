import os
import torch
import numpy as np
import pandas as pd

from torch_geometric.data import Data

from graphxai.utils import Explanation, edge_mask_from_node_mask, aggregate_explanations

#def load_graphs(dir_path: str, smiles_df_path: str):
def load_graphs(datapath: str):
    '''
    Extracts datasets from a format consistent with that used by Sanchez-Lengeling et al., Neurips 2020

    TODO: replace path with Harvard Dataverse loading

    Args:
        dir_path (str): Path to directory containing all graphs
        smiles_df_path (str): Path to CSV file containing all information about SMILES 
            representations of the molecules.

    :rtype: :obj:`(List[torch_geometric.data.Data], List[List[Explanation]], List[int])`
    Returns:
        all_graphs (list of `torch_geometric.data.Data`): List of all graphs in the
            dataset
        explanations (list of `Explanation`): List of all ground-truth explanations for
            each corresponding graph. Ground-truth explanations consist of multiple
            possible explanations as some of the molecular prediction tasks consist
            of multiple possible pathways to predicting a given label.
        zinc_ids (list of ints): Integers that map each molecule back to its original
            ID in the ZINC dataset. 
    '''
    
    # att = np.load(os.path.join(dir_path, 'true_raw_attribution_datadicts.npz'),
    #         allow_pickle = True)
    # X = np.load(os.path.join(dir_path, 'x_true.npz'), allow_pickle = True)
    # y = np.load(os.path.join(dir_path, 'y_true.npz'), allow_pickle = True)
    data = np.load(datapath, allow_pickle = True)
    att, X, y, df = data['attr'], data['X'], data['y'], data['smiles']
    # print(att)
    # print(X)
    # print(df)
    #ylist = [y['y'][i][0] for i in range(y['y'].shape[0])]
    ylist = [y[i][0] for i in range(y.shape[0])]

    #att = att['datadict_list']
    X = X[0]

    #df = pd.read_csv(os.path.join(dir_path, smiles_df_path))

    # Unique zinc identifiers:
    #zinc_ids = df['mol_id'].tolist()
    zinc_ids = df[:,1]

    all_graphs = []
    explanations = []


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
        #print(node_imp.shape)

        # Error-check:
        assert att[i][0]['n_edge'] == X[i]['n_edge'], 'Num: {}, Edges different sizes'.format(i)
        assert node_imp.shape[0] == x.shape[0], 'Num: {}, Shapes: {} vs. {}'.format(i, node_imp.shape[0], x.shape[0]) \
            + '\nExp: {} \nReal:{}'.format(att[i][0], X[i])

        i_exps = []

        for j in range(node_imp.shape[1]):

            exp = Explanation(
                feature_imp = None, # No feature importance - everything is one-hot encoded
                node_imp = node_imp[:,j],
                edge_imp = edge_mask_from_node_mask(node_imp[:,j].bool(), edge_index = edge_index),
            )
            
            exp.set_whole_graph(data_i)
            exp.has_match = (torch.sum(node_imp[:,j] > 0).item() > 0)
            i_exps.append(exp)
            
        explanations.append(i_exps)

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

    aggregate_explanations(exp[i], node_level = False).visualize_graph(show = True)
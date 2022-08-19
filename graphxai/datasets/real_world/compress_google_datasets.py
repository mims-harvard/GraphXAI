import os
import numpy as np
import pandas as pd

def save_graphs(dir_path: str, smiles_df_path: str, to_save: str):
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
    
    att = np.load(os.path.join(dir_path, 'true_raw_attribution_datadicts.npz'),
            allow_pickle = True)
    X = np.load(os.path.join(dir_path, 'x_true.npz'), allow_pickle = True)
    y = np.load(os.path.join(dir_path, 'y_true.npz'), allow_pickle = True)
    ylist = [y['y'][i][0] for i in range(y['y'].shape[0])]

    att, X, y = att['datadict_list'], X['datadict_list'], y['y']

    df = pd.read_csv(os.path.join(dir_path, smiles_df_path))

    np.savez(to_save, attr=att, X=X, y=y, smiles=df)

if __name__ == '__main__':

    save_graphs(dir_path = 'benzene/benzene_data', 
            smiles_df_path='benzene_smiles.csv',  
            to_save = 'benzene/benzene.npz')

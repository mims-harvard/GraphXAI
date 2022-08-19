import os
import torch

from graphxai.datasets.real_world.extract_google_datasets import load_graphs 
from graphxai.datasets import GraphDataset


ATOM_TYPES = [
    'C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'Na', 'Ca', 'I', 'B', 'H', '*'
]

fc_data_dir = os.path.join(os.path.dirname(__file__), 'fc_data')
fc_smiles_df = 'FC_smiles.csv'

fc_datapath = os.path.join(os.path.dirname(__file__), 'fluoride_carbonyl.npz')

class FluorideCarbonyl(GraphDataset):

    def __init__(
            self,
            split_sizes = (0.7, 0.2, 0.1),
            seed = None,
            data_path: str = fc_datapath,
            device = None,
        ):
        '''
        Args:
            split_sizes (tuple): 
            seed (int, optional):
            data_path (str, optional):
        '''
        
        self.device = device

        # self.graphs, self.explanations, self.zinc_ids = \
        #     load_graphs(data_path, os.path.join(data_path, fc_smiles_df))

        self.graphs, self.explanations, self.zinc_ids = load_graphs(data_path)

        super().__init__(name = 'FluorideCarbonyl', seed = seed, split_sizes = split_sizes, device = device)
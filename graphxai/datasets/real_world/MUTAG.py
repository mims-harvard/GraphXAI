import torch

from torch_geometric.datasets import TUDataset

from graphxai.datasets.dataset import GraphDataset
from graphxai.datasets.utils.substruct_chem_match import match_NO2


class GraphXAI_MUTAG(GraphDataset):
    '''
    GraphXAI implementation MUTAG dataset
        - Contains MUTAG with ground-truth 

    Args:
        root (str): Root directory in which to store the dataset
            locally.
        generate (bool, optional): (:default: :obj:`False`) 
    '''

    def __init__(self,
        root: str,
        use_fixed_split: bool = True, 
        generate: bool = True
        ):
        super().__init__(name = 'MUTAG')

        self.dataset = TUDataset(root=root, name='MUTAG')

        # Qualitative variables of dataset:
        self.num_node_features = self.dataset.num_node_features
        self.num_edge_features = self.dataset.num_edge_features




    def __make_explanations(self):
        '''
        Makes explanations for MUTAG dataset
        '''

        # Need to do substructure matching

        pass

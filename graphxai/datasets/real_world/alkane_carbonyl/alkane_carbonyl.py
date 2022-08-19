import os, random
from graphxai.utils import Explanation
import torch

from graphxai.datasets.real_world.extract_google_datasets import load_graphs 
from graphxai.datasets import GraphDataset


ATOM_TYPES = [
    'C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'Na', 'Ca', 'I', 'B', 'H', '*'
]

# fc_data_dir = os.path.join(os.path.dirname(__file__), 'ac_data')
# fc_smiles_df = 'AC_smiles.csv'

ac_datapath = os.path.join(os.path.dirname(__file__), 'alkane_carbonyl.npz')

class AlkaneCarbonyl(GraphDataset):

    def __init__(
            self,
            split_sizes = (0.7, 0.2, 0.1),
            seed = None,
            data_path: str = ac_datapath,
            device = None,
            downsample = True,
            downsample_seed = None
        ):
        '''
        Args:
            split_sizes (tuple): 
            seed (int, optional):
            data_path (str, optional):
        '''
        
        self.device = device
        self.downsample = downsample
        self.downsample_seed = downsample_seed

        self.graphs, self.explanations, self.zinc_ids = load_graphs(data_path)

        # Downsample because of extreme imbalance:
        yvals = [self.graphs[i].y for i in range(len(self.graphs))]

        zero_bin = []
        one_bin = []

        if downsample:
            for i in range(len(self.graphs)):
                if self.graphs[i].y == 0:
                    zero_bin.append(i)
                else:
                    one_bin.append(i)

            # Sample down to keep the dataset balanced
            random.seed(downsample_seed)
            keep_inds = random.sample(zero_bin, k = 2 * len(one_bin))

            self.graphs = [self.graphs[i] for i in (keep_inds + one_bin)]
            self.explanations = [self.explanations[i] for i in (keep_inds + one_bin)]
            self.zinc_ids = [self.zinc_ids[i] for i in (keep_inds + one_bin)]

        super().__init__(name = 'AklaneCarbonyl', seed = seed, split_sizes = split_sizes, device = device)
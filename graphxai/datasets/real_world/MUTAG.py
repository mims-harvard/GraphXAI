import torch

from torch_geometric.datasets import TUDataset

from graphxai.datasets.dataset import GraphDataset
from graphxai.utils import Explanation
from graphxai.datasets.utils.substruct_chem_match import match_NH2, match_substruct, MUTAG_NO2


class MUTAG(GraphDataset):
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

        self.graphs = TUDataset(root=root, name='MUTAG')
        # Dataset retains all qualitative and quantitative attributes from PyG

        # # Qualitative variables of dataset:
        # self.num_node_features = self.dataset.num_node_features
        # self.num_edge_features = self.dataset.num_edge_features

        self.__make_explanations()



    def __make_explanations(self):
        '''
        Makes explanations for MUTAG dataset
        '''

        self.explanations = []

        # Need to do substructure matching
        for i in range(len(self.graphs)):
            print('-------------------------------------------')
            print('Graph {}'.format(i))

            molG = self.get_graph_as_networkx(i)

            node_imp = torch.zeros(molG.number_of_nodes())

            #print(molG.nodes(data=True))

            # Screen for NH2:
            for n in molG.nodes():

                # Screen all nodes through match_NH2
                # match_NH2 is very quick
                if match_NH2(molG, n):
                    print('match')

                # If NH2 match, mask-in the nodes

            # Screen for NO2:
            no2_matches = match_substruct(molG, MUTAG_NO2)
            
            for m in no2_matches:
                node_imp[m] = 1 # Mask-in those values

            # TODO: mask-in edge importance

            exp = Explanation(
                node_imp = node_imp
            )

            exp.set_whole_graph(self.graphs[i].x, self.graphs[i].edge_index)

            self.explanations.append(exp)

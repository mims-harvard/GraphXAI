import torch

from sklearn.model_selection import train_test_split

from torch_geometric.datasets import TUDataset

from graphxai.datasets.dataset import GraphDataset
from graphxai.utils import Explanation, match_edge_presence
from graphxai.datasets.utils.substruct_chem_match import match_NH2, match_substruct_mutagenicity, MUTAG_NO2, MUTAG_NH2

# 0	C
# 1	O
# 2	Cl
# 3	H
# 4	N
# 5	F
# 6	Br
# 7	S
# 8	P
# 9	I
# 10	Na
# 11	K
# 12	Li
# 13	Ca


class Mutagenicity(GraphDataset):
    '''
    GraphXAI implementation Mutagenicity dataset
        - Contains Mutagenicity with ground-truth 

    Args:
        root (str): Root directory in which to store the dataset
            locally.
        generate (bool, optional): (:default: :obj:`False`) 
    '''

    def __init__(self,
        root: str,
        use_fixed_split: bool = True, 
        generate: bool = True,
        split_sizes = (0.7, 0.2, 0.1),
        seed = None
        ):

        self.graphs = TUDataset(root=root, name='Mutagenicity')
        # self.graphs retains all qualitative and quantitative attributes from PyG

        self.__make_explanations()

        super().__init__(name = 'Mutagenicity', seed = seed, split_sizes = split_sizes)


    def __make_explanations(self):
        '''
        Makes explanations for Mutagenicity dataset
        '''

        self.explanations = []

        # Need to do substructure matching
        for i in range(len(self.graphs)):

            molG = self.get_graph_as_networkx(i)

            # Screen for NH2:
            nh2_matches = match_substruct_mutagenicity(molG, MUTAG_NH2, nh2_no2 = 0)

            # Screen for NO2:
            no2_matches = match_substruct_mutagenicity(molG, MUTAG_NO2, nh2_no2 = 1)

            all_matches = nh2_matches + no2_matches # Combine lists

            eidx = self.graphs[i].edge_index

            explanations_i = []

            for m in all_matches:
                node_imp = torch.zeros((molG.number_of_nodes(),))
                
                node_imp[m] = 1
                edge_imp = match_edge_presence(eidx, m)

                exp = Explanation(
                    node_imp = node_imp.float(),
                    edge_imp = edge_imp.float()
                )

                exp.set_whole_graph(self.graphs[i])

                exp.has_match = True

                explanations_i.append(exp)

            if len(explanations_i) == 0:
                # Set a null explanation:
                exp = Explanation(
                    node_imp = torch.zeros((molG.number_of_nodes(),), dtype = torch.float),
                    edge_imp = torch.zeros((eidx.shape[1],), dtype = torch.float)
                )

                exp.set_whole_graph(self.graphs[i])

                exp.has_match = False

                explanations_i = [exp]

            self.explanations.append(explanations_i)
            
            # cumulative_edge_mask = torch.zeros(eidx.shape[1]).bool()
            
            # for m in no2_matches:
            #     node_imp[m] = 1 # Mask-in those values

            #     # Update edge mask:
                
            #     cumulative_edge_mask = cumulative_edge_mask.bool() | (match_edge_presence(eidx, m))

            # for m in nh2_matches:
            #     node_imp[m] = 1

            #     # Update edge_mask:
            
            #     cumulative_edge_mask = cumulative_edge_mask.bool() | (match_edge_presence(eidx, m))

            # exp = Explanation(
            #     node_imp = node_imp,
            #     edge_imp = cumulative_edge_mask.float(),
            # )

            # exp.set_whole_graph(self.graphs[i])

            # self.explanations.append(exp)

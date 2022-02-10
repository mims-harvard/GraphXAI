import torch
import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx, remove_isolated_nodes

from graphxai.datasets.dataset import GraphDataset
from graphxai.utils import Explanation, match_edge_presence
from graphxai.datasets.utils.substruct_chem_match import match_NH2, match_substruct_mutagenicity, MUTAG_NO2, MUTAG_NH2
from graphxai.datasets.utils.substruct_chem_match import match_aliphatic_halide, match_nitroso, match_azo_type, match_polycyclic
from graphxai.utils import aggregate_explanations

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

def make_iter_combinations(length):
    '''
    Builds increasing level of combinations, including all comb's at r = 1, ..., length - 1
    Used for building combinations of explanations
    '''

    if length == 1:
        return [[0]]

    inds = np.arange(length)

    exps = [[i] for i in inds]
    
    for l in range(1, length - 1):
        exps += list(itertools.combinations(inds, l + 1))

    exps.append(list(inds)) # All explanations

    return exps


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
        seed = None,
        test_debug = False,
        device = None,
        ):

        self.device = device

        self.graphs = list(TUDataset(root=root, name='Mutagenicity'))
        # self.graphs retains all qualitative and quantitative attributes from PyG

        # Remove isolated nodes:
        for i in range(len(self.graphs)):
            edge_idx, _, node_mask = remove_isolated_nodes(self.graphs[i].edge_index, num_nodes = self.graphs[i].x.shape[0])
            self.graphs[i].x = self.graphs[i].x[node_mask]
            #print('Shape x', self.graphs[i].x.shape)
            self.graphs[i].edge_index = edge_idx

        self.__make_explanations(test_debug)

        # Filter based on label-explanation validity:
        self.__filter_dataset()

        super().__init__(name = 'Mutagenicity', seed = seed, split_sizes = split_sizes, device = device)


    def __make_explanations(self, test: bool = False):
        '''
        Makes explanations for Mutagenicity dataset
        '''

        self.explanations = []

        # Testing
        if test:
            count_nh2 = 0
            count_no2 = 0
            count_halide = 0
            count_nitroso = 0
            count_azo_type = 0

        # Need to do substructure matching
        for i in range(len(self.graphs)):

            molG = self.get_graph_as_networkx(i)

            if test:
                if molG.number_of_nodes() != self.graphs[i].x.shape[0]:
                    print('idx', i)
                    print('from data', self.graphs[i].x.shape)
                    print('from molG', molG.number_of_nodes())
                    print('edge index unique:', torch.unique(self.graphs[i].edge_index).tolist())
                    tmpG = to_networkx(self.graphs[i], to_undirected=True)
                    print('From PyG nx graph', tmpG.number_of_nodes())

            # Screen for NH2:
            nh2_matches = match_substruct_mutagenicity(molG, MUTAG_NH2, nh2_no2 = 0)

            # Screen for NO2:
            no2_matches = match_substruct_mutagenicity(molG, MUTAG_NO2, nh2_no2 = 1)

            # Screen for aliphatic halide
            halide_matches = match_aliphatic_halide(molG)

            # Screen for nitroso
            nitroso_matches = match_nitroso(molG)

            # Screen for azo-type
            azo_matches = match_azo_type(molG)

            if test:
                count_nh2 += int(len(nh2_matches) > 0)
                count_no2 += int(len(no2_matches) > 0)
                count_halide += int(len(halide_matches) > 0)
                count_nitroso += int(len(nitroso_matches) > 0)
                count_azo_type += int(len(azo_matches) > 0)

            all_matches = nh2_matches + no2_matches + nitroso_matches + azo_matches + halide_matches 

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
            
            else:
                # Combinatorial combination of matches:
                exp_matches_inds = make_iter_combinations(len(all_matches))

                comb_explanations = []

                # Run combinatorial build of all explanations
                for eid in exp_matches_inds:
                    # Get list of explanations:
                    L = [explanations_i[j] for j in eid]
                    tmp_exp = aggregate_explanations(L, node_level = False)
                    tmp_exp.has_match = True
                    comb_explanations.append(tmp_exp) # No reference provided
                    

                self.explanations.append(comb_explanations)

        if test:
            print(f'NH2: {count_nh2}')
            print(f'NO2: {count_no2}')
            print(f'Halide: {count_halide}')
            print(f'Nitroso: {count_nitroso}')
            print(f'Azo-type: {count_azo_type}')
            #print(f'Poly: {count_poly}')
            
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

    def __filter_dataset(self):
        '''
        TODO: could merge this function into __make_explanations, easier to keep
            it here for now
        '''
        #self.label_exp_mask = torch.zeros(len(self.graphs), dtype = bool)

        new_graphs = []
        new_exps = []

        for i in range(len(self.graphs)):
            matches = int(self.explanations[i][0].has_match)
            yval = int(self.graphs[i].y.item())

            #self.label_exp_mask[i] = (matches == yval)
            if matches == yval:
                new_graphs.append(self.graphs[i])
                new_exps.append(self.explanations[i])

        # Perform filtering:
        #self.graphs = [self.graphs[] for i in self.label_exp_mask]
        self.graphs = new_graphs
        self.explanations = new_exps
        # mask_inds = self.label_exp_mask.nonzero(as_tuple = True)[0]
        # self.explanations = [self.explanations[i.item()] for i in mask_inds]



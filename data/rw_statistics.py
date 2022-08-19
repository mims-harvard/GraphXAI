import os, sys; sys.path.append('..')
import numpy as np
from torch_geometric.datasets import TUDataset

from graphxai.explainers import IntegratedGradExplainer
#from graphxai.explainers.utils.visualizations import visualize_mol_explanation
from graphxai.gnn_models.graph_classification import train, test
from graphxai.gnn_models.graph_classification.gcn import GCN_2layer, GCN_3layer
from graphxai.gnn_models.graph_classification.gin import GIN_2layer, GIN_3layer
from graphxai.datasets import Benzene, AlkaneCarbonyl, FluorideCarbonyl, Mutagenicity

# np.std(level_values) / np.sqrt(len(level_values))

default_root = './RW'
def get_dataset(name, root = default_root, device = None):
    '''
    Args:
        root: Only needed if using Mutagenicity
    '''

    if name.lower() == 'benzene':
        return Benzene(split_sizes = (0.7,0.15,0.15), seed = 1234, device = device)

    elif (name.lower() == 'fc') or (name.lower() == 'fluoridecarbonyl'):
        return FluorideCarbonyl(split_sizes = (0.7, 0.15, 0.15), seed = 1234, device = device)

    elif (name.lower() == 'mutagenicity') or (name.lower() == 'mutag'): 
        return Mutagenicity(root = os.path.join(root, 'data'), split_sizes = (0.7, 0.15, 0.15), seed = 1234, device = device)

    elif (name.lower() == 'ac') or (name.lower() == 'alkanecarbonyl'):
        return AlkaneCarbonyl(split_sizes = (0.7, 0.15, 0.15), seed = 1234, device = device, downsample = False)


# Load data: ------------------------------------------
#dataset = Benzene(split_sizes = (0.8, 0.2, 0), seed = seed)
dataset = get_dataset('ac', device = "cpu")
# Get size of dataset:
print('Num. Graphs: {}'.format(len(dataset)))
# Calc. average vals
nodes, edges, feat = [], [], []
for G, E in dataset:
    nodes.append(G.x.shape[0])
    edges.append(G.edge_index.shape[1])
    feat.append(G.x.shape[1])
print('Num. nodes:  {:.4f} +- {:.4f}'.format(np.mean(nodes), np.std(nodes) / np.sqrt(len(nodes))))
print('Num. edges:  {:.4f} +- {:.4f}'.format(np.mean(edges), np.std(edges) / np.sqrt(len(edges))))
print('Num. feats:  {:.4f} +- {:.4f}'.format(np.mean(feat), np.std(feat) / np.sqrt(len(feat))))

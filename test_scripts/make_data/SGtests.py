import torch
import os
from graphxai.datasets.load_synthetic import load_ShapeGGen
from graphxai.datasets import ShapeGGen
from graphxai.utils.explanation import Explanation
from graphxai.utils.nx_conversion import khop_subgraph_nx
from torch_geometric.utils import k_hop_subgraph

#SG = load_ShapeGGen('data/ShapeGGen/new_unzipped/SG_homophilic.pickle', root = '/Users/owenqueen/Desktop/HMS_research/graphxai_project/GraphXAI')
base = '/Users/owenqueen/Desktop/HMS_research/graphxai_project/GraphXAI/data/ShapeGGen/new_unzipped'
sg = load_ShapeGGen('SG_homophilic.pickle', root = base)

#print(SG.seed)

#k = vars(SG).keys()

#l = {ki:vars(SG)[ki] for ki in k if not (isinstance(vars(SG)[ki], torch.Tensor) or isinstance(vars(SG)[ki], list))}
#print(l)

kwargs = {
    'base_graph': 'ba',
    'verify':False,
    'max_tries_verification':5,
    'homophily_coef':1,
    'seed':1456,
    'shape_method':'house',
    'sens_attribution_noise':0.25,
    'num_hops':3,
    'model_layers':3,
    'make_explanations':True,
    'variant':1,
    'num_subgraphs':100,
    'prob_connection':0.06,
    'subgraph_size':11,
    # Features:
    'class_sep':0.6,
    'n_features':10,
    'n_clusters_per_class':2,
    'n_informative':4,
    'add_sensitive_feature':True,
    'attribute_sensitive_feature':False,
}

#sg = ShapeGGen(**kwargs)

print('loaded')
print('x size', sg.graph.x.shape[0])

count = 0
for i in range(sg.graph.x.shape[0]):
    gt = sg.explanations[i][0].node_imp.shape[0]
    # Get a subgraph around node i:
    nodes, _, _, _ = k_hop_subgraph(i, num_hops = 3, edge_index = sg.graph.edge_index, relabel_nodes=True)
    graph_size = nodes.shape[0]

    if graph_size != gt:
        print('Mismatch, GS vs. GT: {} vs. {}'.format(graph_size, gt))
        count += 1

print('Pct mismatch: {}'.format(count / sg.graph.x.shape[0]))


#sg.dump(os.path.join(base, 'SG_retry.pickle'))
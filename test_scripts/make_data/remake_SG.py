import torch
import os
from graphxai.datasets.load_synthetic import load_ShapeGGen
from graphxai.datasets import ShapeGGen
from graphxai.utils.explanation import Explanation

#SG = load_ShapeGGen('data/ShapeGGen/new_unzipped/SG_homophilic.pickle', root = '/Users/owenqueen/Desktop/HMS_research/graphxai_project/GraphXAI')
base = '/Users/owenqueen/Desktop/HMS_research/graphxai_project/GraphXAI/data/ShapeGGen/new_unzipped'

#print(SG.seed)

#k = vars(SG).keys()

#l = {ki:vars(SG)[ki] for ki in k if not (isinstance(vars(SG)[ki], torch.Tensor) or isinstance(vars(SG)[ki], list))}
#print(l)

kwargs = {
    'base_graph': 'ba',
    'verify':True,
    'max_tries_verification':5,
    'homophily_coef':1,
    'seed':1456,
    'shape_method':'house',
    'sens_attribution_noise':0.25,
    'num_hops':3,
    'model_layers':3,
    'make_explanations':True,
    'variant':1,
    'num_subgraphs':1200,
    'prob_connection':0.006,
    'subgraph_size':11,
    # Features:
    'class_sep':0.6,
    'n_features':10,
    'n_clusters_per_class':2,
    'n_informative':4,
    'add_sensitive_feature':True,
    'attribute_sensitive_feature':False,
}

sg = ShapeGGen(**kwargs)


sg.dump(os.path.join(base, 'SG_retry.pickle'))
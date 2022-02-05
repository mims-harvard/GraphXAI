# Gets the test indices for a given SG dataset
# Saves them so we can parallelize metrics
import os
import torch
import numpy as np

my_base_graphxai = '/home/owq978/GraphXAI'

bah = torch.load(open(os.path.join(my_base_graphxai, 'data/ShapeGraph/unzipped/SG_homophilic.pickle'), 'rb'))

data = bah.get_graph(use_fixed_split=True)

test_set = (data.test_mask).nonzero(as_tuple=True)[0]

# Saves the test set to current directory:
torch.save(test_set, open('test_inds_SG_homophilic.pt', 'wb'))

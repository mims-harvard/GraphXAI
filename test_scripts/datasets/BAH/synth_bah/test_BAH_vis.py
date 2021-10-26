import sys
from graphxai.datasets import BAShapes

assert len(sys.argv) == 4, 'usage: python3 test_BAH_vis.py <shape_insert_strategy> <labeling_method> <feature_method>'

assert sys.argv[1] in ['ub', 'lb', 'global', 'bound_12'], "shape_insert_strategy must be in ['ub', 'lb', 'global']"
assert sys.argv[2] in ['e', 'f', 'ef'], "labeling_method must be in ['e', 'f', 'ef']"
assert sys.argv[3] in ['gaussian', 'ns', 'onehot', 'lv'], "feature_method must be in ['gaussian', 'ns', 'onehot']"

pm_conv = {'ub':'neighborhood upper bound', 'lb':'local', 'global':'global', 'bound_12': 'bound_12'}
lm_conv = {'e':'edge', 'f':'feature', 'ef':'edge and feature'}
fm_conv = {'gaussian':'gaussian', 'ns':'network stats', 'onehot':'onehot', 'lv': 'gaussian_lv'}

if sys.argv[1] == 'global':
    num_shapes = 5
elif sys.argv[1] == 'lb':
    num_shapes = 1
elif sys.argv[1] == 'ub' or sys.argv[1] == 'bound_12':
    num_shapes = None

class Hyperparameters:
    num_hops = 2
    n = 30
    m = 1
    num_shapes = num_shapes
    shape_insert_strategy = pm_conv[sys.argv[1]]
    shape_upper_bound = 1
    shape = 'house'
    labeling_method = lm_conv[sys.argv[2]]

hyp = Hyperparameters
args = {key:value for key, value in hyp.__dict__.items() if not key.startswith('__') and not callable(value)}
bah = BAShapes(**args, feature_method = fm_conv[sys.argv[3]])

print(bah.G.nodes(data=True))

bah.visualize(shape_label = True)
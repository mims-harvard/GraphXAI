from graphxai.datasets import BAShapes

class Hyperparameters:
    num_hops = 2
    n = 50
    m = 1
    num_shapes = None
    plant_method = 'neighborhood upper bound'
    shape_upper_bound = 1
    labeling_method = 'edge'

hyp = Hyperparameters
args = {key:value for key, value in hyp.__dict__.items() if not key.startswith('__') and not callable(value)}
bah = BAShapes(**args, feature_method = 'onehot')

bah.visualize()
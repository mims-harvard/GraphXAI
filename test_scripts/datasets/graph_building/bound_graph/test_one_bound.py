import sys
import matplotlib.pyplot as plt

from graphxai.datasets.shape_graph import ShapeGraph

def parse_args():
    arg_dict = {}

    assert len(sys.argv) == 3, 'usage: python3 test_one_ul.py <num_subgraphs> <prob_connection>'

    arg_dict['num_subgraphs'] = int(sys.argv[1])
    arg_dict['prob_connection'] = float(sys.argv[2])

    return arg_dict


args = parse_args()

bah = ShapeGraph(**args, model_layers=3)
data = bah.get_graph()
G = bah.G

# Get degree distribution of G:
degrees = sorted([d for n, d in G.degree()])

plt.hist(degrees, color = 'green')
plt.title('Degree Distribution')
plt.show()

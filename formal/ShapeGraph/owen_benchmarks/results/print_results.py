import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--prefix', type=str, required=True)

args = parser.parse_args()

fname_base = args.prefix + '_GES_'
suffixes = ['edge', 'node', 'feat']

for s in suffixes:
    f = fname_base + s + '.npy'
    arr = np.load(open(f, 'rb'), allow_pickle = True)
    print(s)
    print(arr)
    print('')
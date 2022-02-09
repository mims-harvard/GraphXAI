import os, argparse, glob
import torch
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--path', required = True)
parser.add_argument('--num_splits', default = 50)
parser.add_argument('--path_to_split', default = '../../test_inds_SG_homophilic.pt')
args = parser.parse_args()

# Load in 
split = torch.load(args.path_to_split).numpy()
num_in_split = split.shape[0]

def get_file_nums():
    flist = glob.glob(args.path + '/*')

    saved_nums = []

    for f in flist:
        if f[-4:] != '.npy':
            continue
        n = os.path.abspath(f)
        num = int(n[-9:-4])

        saved_nums.append(num)

    # Need to return set because there may be repeated elements:
    return set(saved_nums)

def count_sym_diff():

    fnums = get_file_nums()

    set_spl = set(list(split))

    not_saved = set_spl.symmetric_difference(fnums)

    # Bin the split:
    bins = np.arange(0, num_in_split, num_in_split // args.num_splits)

    for i in range(bins.shape[0]):
        # Count how many numbers in this split are in not_saved:
        if i < (bins.shape[0] - 1):
            bin_range = [split[j] for j in range(bins[i], bins[i+1])]
        else:
            bin_range = [split[j] for j in range(bins[i], num_in_split)]

        # Counts off the top:
        off_top = 0
        for j in range(len(bin_range) - 1, -1, -1):
            if bin_range[j] in fnums:
                break
            off_top += 1

        # len_in_split = len(bin_range.intersection(not_saved))

        #print('\t Split {}: {}'.format(i, len_in_split))
        print('\t Split {}: {}'.format(i, off_top))

count_sym_diff()

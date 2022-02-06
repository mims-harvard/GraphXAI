import sys; sys.path.append()
import argparse, random
import numpy as np
import torch

from utils import get_model, get_exp_method

parser = argparse.ArgumentParser()
parser.add_argument('--exp_method', required=True, help='name of the explanation method')
parser.add_argument('--stability', action='store_true')
parser.add_argument('--accuracy', action='store_true')
parser.add_argument('--model', required = True, default='GIN', type=str, help = 'name of model for benchmarking')
parser.add_argument('--save_dir', default='./fairness_results/', help='folder for saving results')

seed_value=912
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)

# Get testing set:

import os, glob
import torch
import argparse
import numpy as np

def aggregate_results(exp_method, metric, level, path):
    '''
    Metrics should match:
        {exp_method}_{metric}_{level}

    Also performs filtering for None's

    Assumes they are each dictionaries (standard usage in Owen's parallelization)
    '''
    
    exp_method = exp_method.upper()
    
    flist = glob.glob(os.path.join(path, '{}_{}_{}*.npy'.format(exp_method, metric, level)))

    all_runs = []
    for f in flist:
        d = np.load(open(f, 'rb'), allow_pickle = True).item()

        for _, v in d.items():
            if v is not None:
                all_runs.append(v)

    return all_runs
    
def summarize_metrics(exp_method, metric, path, agg = False):
    
    levels = ['feat', 'node', 'edge']

    exp_method = exp_method.upper()

    for l in levels:

        if agg:
            level_values = aggregate_results(exp_method, metric.upper(), l, path)
        else:
            level_values = np.load(os.path.join(path, \
                '{}_{}_{}.npy'.format(exp_method.upper(), metric.upper(), l)),
                allow_pickle = True)
            
            # Filter for None's:
            level_values = level_values[level_values != np.array(None)]

        if len(level_values) == 0:
            p_str = f'{exp_method} has no {l} values for {metric.upper()}'
        else:
            p_str = f'{exp_method}, {metric.upper()}, {l} Mean: {np.mean(level_values)} '
            p_str += f'+- {np.std(level_values) / np.sqrt(len(level_values))}'

        print(p_str)
        print('')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True, help = 'Path to relevant directory containing results (relative or absolute)')
    parser.add_argument('--exp_method', required=True, help='name of the explanation method')
    parser.add_argument('--metric', required=True, help = 'Metric name: GES, GCF, or GGF')
    parser.add_argument('--agg', action='store_true', help = 'If method was ran in parallel or not, \
            i.e. results need to be aggregated from multiple files.')

    args = parser.parse_args()

    summarize_metrics(args.exp_method, args.metric, args.path, args.agg)


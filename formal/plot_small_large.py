import os
import ipdb
import glob
import numpy as np
import pandas as pd
import seaborn as sea
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 27})
plt.rc('font', family='sans-serif')
plt.rcParams["axes.grid"] = False
plt.rc('font', family='sans-serif')


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


# inits
algos = ['rand', 'grad', 'cam', 'gcam', 'gbp', 'ig', 'gnnex', 'pgmex', 'pgex']
df_small = []
df_large = []
ty = 'node'
# Loop over all datasets
for ind, algo in enumerate(['rand', 'grad', 'cam', 'gcam', 'gbp', 'ig', 'gnnex', 'pgmex', 'pgex']):
    temp_small = np.load(f'./results_small/{algo}_gef_{ty}.npy', allow_pickle=True)
    if temp_small[-1] is not None:
        df_small.append(temp_small)
    else:
        algos.remove(algo)
    temp_large = np.load(f'./results_homophily/{algo}_gef_{ty}.npy', allow_pickle=True)
    if temp_large[-1] is not None:
        df_large.append(temp_large)

# print statistics
for i,val in enumerate(df_small):
    print(f'{algos[i]}  | Mean_small:{np.mean(val)} | Mean_large:{np.mean(df_large[i])}')
exit(0)

 
# plotting distributions
fig, ax = plt.subplots(figsize=(20, len(df_small)))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

sm = ax.boxplot(df_small, positions=np.array(range(len(algos)))/9-0.0125, sym='', widths=0.025, whis=(5, 95), patch_artist=True, meanline=True, showmeans=True)
xnorm = ax.boxplot(df_large, positions=np.array(range(len(algos)))/9+0.0125, sym='', widths=0.025, whis=(5, 95), patch_artist=True, meanline=True, showmeans=True)
set_box_color(sm, '#D58DF8') # colors are from http://colorbrewer2.org/
set_box_color(xnorm, '#820BBB')
for median in sm['means']:
    median.set(color ='k',
               linewidth = 2, linestyle='-')

for median in xnorm['means']:
    median.set(color ='k',
               linewidth = 2, linestyle='-')

# draw temporary red and blue lines and use them to create a legend
plt.plot([], c='#77FF77', label='Triangle Motifs')
plt.plot([], c='#009900', label='House Motifs')
plt.xticks(np.array(range(0, len(algos)))/9, ['Random', 'Grad', 'CAM', 'GradCAM', 'GradBP', 'IG', 'GNNEx', 'PGMEx', 'PGEx'])
ax.set_xlim(-0.05, (len(algos)/9)+0.01)
plt.ylabel('Graph Explanation faithfulness')
plt.savefig(f'./small_large.pdf', bbox_inches='tight')

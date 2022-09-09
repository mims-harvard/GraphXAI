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
df_homo = []
df_hete = []
ty = 'node'
# Loop over all datasets
for ind, algo in enumerate(['rand', 'grad', 'cam', 'gcam', 'gbp', 'ig', 'gnnex', 'pgmex', 'pgex']):
    temp_homo = np.load(f'./results_small_homophily/{algo}_gef_{ty}.npy', allow_pickle=True)
    if temp_homo[-1] is not None:
        df_homo.append(temp_homo)
    else:
        algos.remove(algo)
    temp_hete = np.load(f'./results_small_heterophily/{algo}_gef_{ty}.npy', allow_pickle=True)
    if temp_hete[-1] is not None:
        df_hete.append(temp_hete)
    # print(f'{algo}: Homophily={df_homo[-1].mean()} | Heterophily={df_hete[-1].mean()}')

## print statistics
for i, val in enumerate(df_homo):
    print(f'Explanation method: {algos[i]}, Mean_homo={np.mean(val):.3f}, Mean_heto={np.mean(df_hete[i]):.3f}')

exit(0)
    
# plotting distributions
fig, ax = plt.subplots(figsize=(20, len(df_homo)))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

sm = ax.boxplot(df_homo, positions=np.array(range(len(algos)))/9-0.0125, sym='', widths=0.025, whis=(5, 95), patch_artist=True)
xnorm = ax.boxplot(df_hete, positions=np.array(range(len(algos)))/9+0.0125, sym='', widths=0.025, whis=(5, 95), patch_artist=True)
set_box_color(sm, '#77FF77')
set_box_color(xnorm, '#009900')
for median in sm['medians']:
    median.set(color ='k',
               linewidth = 2, linestyle='-')

for median in xnorm['medians']:
    median.set(color ='k',
               linewidth = 2, linestyle='-')

# draw temporary red and blue lines and use them to create a legend
plt.plot([], c='#77FF77', label='Homophily')
plt.plot([], c='#009900', label='Heterophily')
plt.xticks(np.array(range(0, len(algos)))/9, ['Random', 'Grad', 'CAM', 'GradCAM', 'GradBP', 'IG', 'GNNEx', 'PGMEx', 'PGEx'])
ax.set_xlim(-0.05, (len(algos)/9)+0.013)
plt.ylabel('Graph Explanation faithfulness')
plt.savefig(f'./homo_hetero.pdf', bbox_inches='tight')

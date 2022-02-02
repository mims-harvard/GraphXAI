import os
import ipdb
import glob
import numpy as np
import pandas as pd
import seaborn as sea
import matplotlib.pyplot as plt
# plt.style.use('ggplot')

plt.rcParams.update({'font.size': 36})  # , 'font.weight': 'bold'})
plt.rc('font', family='sans-serif')
plt.rcParams["axes.grid"] = False
plt.rc('font', family='sans-serif')


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


# inits
algos = ['rand', 'grad', 'gcam', 'gbp', 'glime', 'ig']  # , 'gnnex']  # , 'pgmex']
df_small = []
df_large = []
ty = 'node'
# Loop over all datasets
for ind, algo in enumerate(['rand', 'grad', 'gcam', 'gbp', 'glime', 'ig']):  # , 'gnnex']):  # , 'pgmex']):
    temp_small = np.load(f'./results_small/{algo}_gea_{ty}.npy', allow_pickle=True)
    if temp_small[-1] is not None:
        df_small.append(temp_small)
    else:
        algos.remove(algo)
    temp_large = np.load(f'./results_large/{algo}_gea_{ty}.npy', allow_pickle=True)
    if temp_large[-1] is not None:
        df_large.append(temp_large)
    # print(f'{algo}: Small={df_small[-1].mean()} | Large={df_large[-1].mean()}')
# plotting distributions
fig, ax = plt.subplots(figsize=(70, len(df_small)))

# ipdb.set_trace()
sm = ax.boxplot(df_small, positions=np.array(range(len(algos)))*2.0-0.4, sym='', widths=0.6, whis=(5, 95), medianprops=dict(color='k'), patch_artist=True)
xnorm = ax.boxplot(df_large, positions=np.array(range(len(algos)))*2.0+0.4, sym='', widths=0.6, whis=(5, 95), medianprops=dict(color='k'), patch_artist=True)
set_box_color(sm, '#77FF77') # colors are from http://colorbrewer2.org/
set_box_color(xnorm, '#009900')

# draw temporary red and blue lines and use them to create a legend
plt.plot([], c='#77FF77', label='Triangle Motifs')  # Triangle motifs')
plt.plot([], c='#009900', label='House Motifs')  # oause motifs')
plt.xticks(range(0, len(algos) * 2, 2), algos)
# plt.yticks(range(0, 1, 0.1), fontsize=36)  # len(df_small)**2, 10), fontsize=36)
plt.legend(bbox_to_anchor=(0.9, 0.6))
plt.savefig(f'./demo.pdf', bbox_inches='tight')
# plt.savefig(f'./empirical_bound_lg_faithfulness.pdf', bbox_inches='tight')








# my_pal = "tab10"  # {"GCN": "#FF99AD", "GIN": "#9AF8E3", "SAGE": "#FDF19D", "INFOMAX": "#77FF77", "JK": "#D58DF8"}
exit(0)
#if metric == 'Unfairness':
# ax = sea.boxplot(y=f"{metric}", data=df_softmax, linewidth=0.25, palette=my_pal, width=0.63)
#    ax.set_xlabel("", fontsize=fontsize)  # , fontweight='bold')
#    ax.set_ylabel(f"{metric}", fontsize=fontsize)  # , fontweight='bold')
#    plt.yticks([-2, 0, 5, 10, 15, 20, 25], [' ', '0', '5', '10', '15', '20', '25'])
#    plt.xticks([])
#    plt.legend([],[], frameon=False)
#    # ax.axhline(0.0, linestyle='--', color='k', linewidth=2.0)
#    # ax.text(0.03, -2.7, 'Fair Model', fontsize=fontsize)
#elif metric == 'Instability':
#    ax = sea.boxplot(x="Dataset", y=f"{metric}", hue="GNN_Model", data=df, linewidth=0.25, palette=my_pal, width=0.63)
#    ax.set_xlabel("", fontsize=fontsize)  # , fontweight='bold')
#    ax.set_ylabel(f"{metric}", fontsize=fontsize)  # , fontweight='bold')
#    # plt.yticks([-2, 0, 10, 20, 30, 40, 50], [' ', '0', '10', '20', '30', '40', '50'])
#    plt.yticks([-2, 0, 10, 20, 30, 40, 50], [' ', '0', '10', '20', '30', '40', '50'])
#    plt.xticks([])
#    # plt.legend([],[], frameon=False)
#    # plt.legend(bbox_to_anchor=(0.95, 1), loc='upper left', ncol=5)
#    # ax.axhline(0.0, linestyle='--', color='k', linewidth=2.0)
#    # ax.text(0.07, -6.3, 'Stable Model', fontsize=fontsize)
#elif metric == 'both':
#    ax = sea.boxplot(x="Dataset", y="Unfairness", hue="GNN_Model", data=df, linewidth=0.25, palette=my_pal, width=0.45)
#    plt.legend([],[], frameon=False)
#    ax = sea.boxplot(x="Dataset", y="Instability", hue="GNN_Model", data=df, linewidth=0.25, palette=my_pal, width=0.45)
#    ax.set_xlabel("", fontweight='bold')
#    ax.set_ylabel(f"{metric} (error in %)", fontsize=16)  # , fontweight='bold')
#    plt.yticks([-2, 0, 10, 20, 30, 40], [' ', '0', '10', '20', '30', '40'])
#    # ax.axhline(0.0)
#    # ax.text(3.7, -1.5, 'Unfair Line', fontsize=14)
#
#sea.despine(left=True, bottom=True)
## plt.yticks([i/10 for i in range(0, 11, 2)])
#plt.legend(ncol=5)
#plt.savefig(f'./{metric}.pdf', bbox_inches='tight')
## plt.show()
## ipdb.set_trace()
#

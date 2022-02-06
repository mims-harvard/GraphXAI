import os
import ipdb
import glob
import numpy as np
import pandas as pd
import seaborn as sea
import matplotlib.pyplot as plt
# plt.style.use('ggplot')

plt.rcParams.update({'font.size': 27})  # , 'font.weight': 'bold'})
plt.rc('font', family='sans-serif')
plt.rcParams["axes.grid"] = False
plt.rc('font', family='sans-serif')


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


# inits
algos = ['rand', 'grad', 'gcam', 'gbp']  # , 'ig', 'gnnex', 'pgmex', 'pgex']
df_homo = []
df_hete = []
ty = 'node'
# Loop over all datasets
for ind, algo in enumerate(['rand', 'grad', 'gcam', 'gbp']):   # , 'ig', 'gnnex', 'pgmex', 'pgex']):
    temp_homo = np.load(f'./results_homophily/{algo}_gef_{ty}.npy', allow_pickle=True)
    if temp_homo[-1] is not None:
        df_homo.append(temp_homo)
    else:
        algos.remove(algo)
    temp_hete = np.load(f'./results_heterophily/{algo}_gef_{ty}.npy', allow_pickle=True)
    if temp_hete[-1] is not None:
        df_hete.append(temp_hete)
    # print(f'{algo}: Homophily={df_homo[-1].mean()} | Heterophily={df_hete[-1].mean()}')

### print statistics
#for i, val in enumerate(df_homo):
#    print(f'Explanation method: {algos[i]}, MeanGEF={np.mean(val):.3f}, StdGEF={np.std(val):.3f}')

# exit(0)
    
# plotting distributions
fig, ax = plt.subplots(figsize=(20, len(df_homo)))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# ipdb.set_trace()
# my_pal = {"GCN": "#FF99AD", "NIFTY-GCN": "#FF0033", "GIN": "#9AF8E3", "NIFTY-GIN": "#0FDDAF", "SAGE": "#FDF19D", "NIFTY-SAGE": "#FBDB0C", "INFOMAX": "#77FF77", "NIFTY-INFOMAX": "#009900", "JK": "#D58DF8", "NIFTY-JK": "#820BBB"}
sm = ax.boxplot(df_homo, positions=np.array(range(len(algos)))/8-0.0125, sym='', widths=0.025, whis=(5, 95), patch_artist=True, meanline=True, showmeans=True)
xnorm = ax.boxplot(df_hete, positions=np.array(range(len(algos)))/8+0.0125, sym='', widths=0.025, whis=(5, 95), patch_artist=True, meanline=True, showmeans=True)
set_box_color(sm, '#77FF77')  # '#D7191C') # colors are from http://colorbrewer2.org/
set_box_color(xnorm, '#009900')  # '#2C7BB6')
for median in sm['means']:
    median.set(color ='k',
               linewidth = 2, linestyle='-')

for median in xnorm['means']:
    median.set(color ='k',
               linewidth = 2, linestyle='-')

# draw temporary red and blue lines and use them to create a legend
plt.plot([], c='#77FF77', label='Homophily')  # Triangle motifs')
plt.plot([], c='#009900', label='Heterophily')  # oause motifs')
plt.xticks(np.array(range(0, len(algos)))/8, ['Random', 'Grad', 'GradCAM', 'GradBP'])  # , 'IG', 'GNNEx', 'PGMEx', 'PGEx'])
ax.set_xlim(-0.05, (len(algos)/8)+0.013)
# plt.yticks(range(0, 1, 0.1), fontsize=36)  # len(df_small)**2, 10), fontsize=36)
plt.ylabel('Graph Explanation faithfulness')
# plt.legend()  # bbox_to_anchor=(0.9, 0.6))
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

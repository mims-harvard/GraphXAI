import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from graphxai.datasets import ShapeGraph
from test_homophily import homophily_test

def build_SG(HF):

    SG = ShapeGraph(
        model_layers = 3,
        num_subgraphs = 100,
        prob_connection = 0.08,
        subgraph_size = 10,
        seed = 1934,
        class_sep = 1,
        n_informative = 6,
        n_clusters_per_class = 1,
        verify = False,
        make_explanations=False,
        homophily_coef = HF
    )

    con0, ncon0 = homophily_test(SG, label = 0)

    con1, ncon1 = homophily_test(SG, label = 1)

    diff0 = np.mean(con0).item() - np.mean(ncon0).item()
    diff1 = np.mean(con1).item() - np.mean(ncon1).item()

    return diff0, diff1

def vary_HF():
    HF_range = np.linspace(-0.0001, 0.0001, num=20)

    all_diff0 = []
    all_diff1 = []

    for HF in HF_range:
        diff0, diff1 = build_SG(HF)

        all_diff0.append(diff0)
        all_diff1.append(diff1)

    plt.plot(HF_range, all_diff0, label='Label 0')
    plt.plot(HF_range, all_diff1, label='Label 1')
    plt.xlabel('HF')
    plt.ylabel('Connected - Disconnected')
    #plt.xticks(np.log10(HF_range), ['{:2e}'.format(n) for n in np.log10(HF_range)])
    plt.title('Cos Sim. of Connected vs. Disconnected Nodes')
    plt.show()

if __name__ == '__main__':
    vary_HF()
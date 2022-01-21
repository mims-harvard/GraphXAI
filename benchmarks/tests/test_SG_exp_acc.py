import torch
import matplotlib.pyplot as plt
from tqdm import trange

from graphxai.datasets import ShapeGraph
from graphxai.explainers import GNNExplainer, GradExplainer
from graphxai.metrics import graph_exp_acc, graph_exp_faith, graph_exp_stability
from graphxai.gnn_models.node_classification.testing import GCN_3layer_basic, GIN_3layer_basic, test, train 

from graphxai.utils import correct_predictions

def experiment():
    SG = ShapeGraph(model_layers = 3, 
        num_subgraphs = 75, 
        prob_connection = 0.08, 
        subgraph_size = 8)
    data = SG.get_graph(use_fixed_split=True)

    # Test on 3-layer basic GCN, 16 hidden dim:
    model = GCN_3layer_basic(16, input_feat = 11, classes = 2)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, 101):
        loss = train(model, optimizer, criterion, data)

    f1, acc, precision, recall, auroc, auprc = test(model, data, get_auc = True)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test F1: {f1:.4f}, Test AUROC: {auroc:.4f}')

    model.eval()

    #gnnexp = GradExplainer(model, criterion = criterion)
    gnnexp = GNNExplainer(model)

    to_test = correct_predictions(model, data, data_mask = data.test_mask, get_mask = False)

    print(to_test)

    accuracies = []
    faiths = []

    for i in trange(to_test.shape[0]):
        # Iterate over nodes where the prediction was correct
        n = to_test[i]

        exp = gnnexp.get_explanation_node(
                    node_idx = n.item(), 
                    x = data.x,  
                    edge_index = data.edge_index)

        gt_exp = SG.explanations[n.item()]

        # Apply top-k importance
        #exp.top_k_node_imp(top_k = 11, inplace=True)
        #exp.top_k_edge_imp(top_k = 12, inplace = True)

        acc = graph_exp_acc(gt_exp, exp)
        faith = graph_exp_faith(exp, SG, model)

        accuracies.append(acc)
        faiths.append(faith)

    return accuracies, faiths

if __name__ == '__main__':

    accuracies, faiths = experiment()

    plt.hist(accuracies)

    plt.show()

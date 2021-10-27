import random, sys
from networkx.classes.function import to_undirected
import networkx as nx
import torch
import matplotlib.pyplot as plt

from torch_geometric.utils import to_networkx

from graphxai.explainers import CAM, Grad_CAM
#from graphxai.explainers.utils.visualizations import visualize_subgraph_explanation
from graphxai.visualization.visualizations import visualize_subgraph_explanation
from graphxai.visualization.explanation_vis import visualize_node_explanation
from graphxai.gnn_models.node_classification import GCN, train, test
from graphxai.gnn_models.node_classification.testing import GCN_3layer_basic
from graphxai.datasets import BAShapes

from graphxai.utils import to_networkx_conv, Explanation

def comp_gt(gt_exp: Explanation, generated_exp: Explanation):
    '''
    Args:
        gt_exp (Explanation): Ground truth explanation from the dataset
        generated_exp: Explanation predicted by an explainer
    '''
    relative_positives = (gt_exp.node_imp == 1).nonzero(as_tuple=True)[0]
    true_nodes = [gt_exp.enc_subgraph.nodes[i].item() for i in relative_positives]

    # Accessing the enclosing subgraph. Will be the same for both explanation.:
    exp_subgraph = generated_exp.enc_subgraph

    calc_node_imp = generated_exp.node_imp

    TPs = []
    FPs = []
    FNs = []
    for i, node in enumerate(exp_subgraph.nodes):
        # Restore original node numbering
        positive = calc_node_imp[i].item() > 0.8
        # print(calc_node_imp[i].item())
        if positive:
            if node in true_nodes:
                TPs.append(node)
            else:
                FPs.append(node)
        else:
            if node in true_nodes:
                FNs.append(node)
    TP = len(TPs)
    FP = len(FPs)
    FN = len(FNs)
    edge_score = TP / (TP + FP + FN + 1e-9)
    print(f'TP / (TP+FP+FN) edge score of gnn explainer is {edge_score}')

    return edge_score

if __name__ == '__main__':

    n = 300
    m = 1
    num_houses = 20

    assert len(sys.argv) == 2, 'usage: python basic_test_CAM.py <difficulty rank of dataset 1-3>'

    if int(sys.argv[1]) == 1:
        hyp = {
            'num_hops': 2,
            'n': n,
            'm': m,
            'num_shapes': num_houses,
            'shape_insert_strategy': 'bound_12',
            'labeling_method': 'edge',
            'shape_upper_bound': 1,
            'feature_method': 'gaussian_lv'
        }
    elif int(sys.argv[1]) == 2:
        hyp = {
            'num_hops': 2,
            'n': n,
            'm': m,
            'num_shapes': num_houses,
            'shape_insert_strategy': 'neighborhood upper bound',
            'labeling_method': 'edge',
            'shape_upper_bound': 1,
            'feature_method': 'gaussian_lv'
        }
    elif int(sys.argv[1]) == 3:
        pass
    else:
        raise NotImplementedError('Must be between 1-3 (ints) for difficulty')

    bah = BAShapes(**hyp)
    data = bah.get_graph(use_fixed_split=True)
    inhouse = (data.y == 0).nonzero(as_tuple=True)[0]

    model = GCN_3layer_basic(64, input_feat = 10, classes = 2)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # Train the model for 200 epochs:
    for epoch in range(1, 201):
        loss = train(model, optimizer, criterion, data)
        acc = test(model, data)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Acc: {acc:.4f}')

    node_idx = random.choice(inhouse)

    model.eval()
    pred = model(data.x, data.edge_index)[node_idx, :].reshape(-1, 1)
    pred_class = pred.argmax(dim=0).item()

    print('GROUND TRUTH LABEL: \t {}'.format(data.y[node_idx].item()))
    print('PREDICTED LABEL   : \t {}'.format(pred.argmax(dim=0).item()))

    act = lambda x: torch.argmax(x, dim=1)
    cam = CAM(model, activation = act)
    cam_exp = cam.get_explanation_node(data.x, node_idx = int(node_idx), label = pred_class, edge_index = data.edge_index)

    gt_exp = bah.explanations[node_idx]

    gcam = Grad_CAM(model, criterion = criterion)
    gcam_exp = gcam.get_explanation_node(
                        data.x, 
                        y = data.y, 
                        node_idx = int(node_idx), 
                        edge_index = data.edge_index, 
                        average_variant=True)

    # Normalize the explanations to 0-1 range:
    cam_exp.node_imp = cam_exp.node_imp / torch.max(cam_exp.node_imp)
    gcam_exp.node_imp = gcam_exp.node_imp / torch.max(gcam_exp.node_imp)

    # Compute Scores
    cam_exp_score = comp_gt(gt_exp, cam_exp)
    gcam_exp_score = comp_gt(gt_exp, gcam_exp)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    gt_exp.context_draw(num_hops = bah.num_hops, additional_hops = 2, heat_by_exp = True, ax = ax1)
    ax1.set_title('Ground Truth')

    cam_exp.context_draw(num_hops = bah.num_hops, additional_hops = 2, heat_by_exp = True, ax = ax2)
    ax2.set_title('CAM')
    ymin, ymax = ax2.get_ylim()
    xmin, xmax = ax2.get_xlim()
    ax2.text(xmin, ymin, 'Faithfulness: {:.3f}'.format(cam_exp_score))

    gcam_exp.context_draw(num_hops = bah.num_hops, additional_hops = 2, heat_by_exp = True, ax = ax3)
    ax3.set_title('Grad-CAM')
    ymin, ymax = ax3.get_ylim()
    xmin, xmax = ax3.get_xlim()
    ax3.text(xmin, ymin, 'Faithfulness: {:.3f}'.format(gcam_exp_score))

    ymin, ymax = ax1.get_ylim()
    xmin, xmax = ax1.get_xlim()
    ax1.text(xmin, ymax - 0.1*(ymax-ymin), 'Label = {:d}'.format(data.y[node_idx].item()))
    ax1.text(xmin, ymax - 0.15*(ymax-ymin), 'Pred  = {:d}'.format(pred.argmax(dim=0).item()))

    plt.show()


import random, sys
from networkx.classes.function import to_undirected
import networkx as nx
import torch
import matplotlib.pyplot as plt

from torch_geometric.utils import to_networkx

from graphxai.explainers import CAM, GradCAM
#from graphxai.explainers.utils.visualizations import visualize_subgraph_explanation
from graphxai.visualization.visualizations import visualize_subgraph_explanation
from graphxai.visualization.explanation_vis import visualize_node_explanation
from graphxai.gnn_models.node_classification import GCN, train, test
from graphxai.gnn_models.node_classification.testing import GCN_3layer_basic
#from graphxai.datasets import BAShapes
from old.new_BAshapes import ShapeGraph

from graphxai.utils import to_networkx_conv, Explanation

def comp_gt(gt_exp: Explanation, generated_exp: Explanation) -> float:
    '''
    Args:
        gt_exp (Explanation): Ground truth explanation from the dataset.
        generated_exp (Explanation): Explanation output by an explainer.
    '''

    # TODO: Check that subgraphs match

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

    hyp = {
        'num_hops': 2,
        'n': n,
        'm': m,
        'num_shapes': num_houses,
        'model_layers': 3,
        'shape_insert_strategy': 'bound_12',
        'labeling_method': 'edge',
        'shape_upper_bound': 1,
        'feature_method': 'gaussian_lv'
    }

    #bah = BAShapes(**hyp)
    bah = ShapeGraph(**hyp)
    data = bah.get_graph(use_fixed_split=True)

    model = GCN_3layer_basic(64, input_feat = 10, classes = 2)
    print(model)
    print('Samples in Class 0', torch.sum(data.y == 0).item())
    print('Samples in Class 1', torch.sum(data.y == 1).item())

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # Train the model for 200 epochs:
    for epoch in range(1, 201):
        loss = train(model, optimizer, criterion, data)
        acc = test(model, data)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Acc: {acc:.4f}')

    model.eval()
    preds = model(data.x, data.edge_index)

    # Choose a node that's in class 1 (connected to 2 houses)
    class1 = (data.y == 1).nonzero(as_tuple=True)[0]
    node_idx = random.choice(class1).item()

    pred = preds[node_idx,:].reshape(-1,1)
    pred_class = pred.argmax(dim=0).item()

    print('GROUND TRUTH LABEL: \t {}'.format(data.y[node_idx].item()))
    print('PREDICTED LABEL   : \t {}'.format(pred.argmax(dim=0).item()))


    # Compute CAM explanation
    act = lambda x: torch.argmax(x, dim=1)
    cam = CAM(model, activation = act)
    cam_exp = cam.get_explanation_node(
        data.x, 
        node_idx = int(node_idx), 
        label = pred_class, 
        edge_index = data.edge_index)
        
    #Explanation
    # Get ground-truth explanation
    gt_exp = bah.explanations[node_idx]

    # Compute Grad-CAM explanation
    gcam = GradCAM(model, criterion = criterion)
    print('node_idx', node_idx)
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

    # Plotting:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    gt_exp.visualize_node(num_hops = bah.num_hops, graph_data = data, additional_hops = 1, heat_by_exp = True, ax = ax1)
    ax1.set_title('Ground Truth')

    cam_exp.visualize_node(num_hops = bah.num_hops, graph_data = data, additional_hops = 1, heat_by_exp = True, ax = ax2)

    ax2.set_title('CAM')
    ymin, ymax = ax2.get_ylim()
    xmin, xmax = ax2.get_xlim()
    ax2.text(xmin, ymin, 'Faithfulness: {:.3f}'.format(cam_exp_score))
    gcam_exp.visualize_node(num_hops = bah.num_hops, graph_data = data, additional_hops = 1, heat_by_exp = True, ax = ax3)
    ax3.set_title('Grad-CAM')
    ymin, ymax = ax3.get_ylim()
    xmin, xmax = ax3.get_xlim()
    ax3.text(xmin, ymin, 'Faithfulness: {:.3f}'.format(gcam_exp_score))

    ymin, ymax = ax1.get_ylim()
    xmin, xmax = ax1.get_xlim()
    print('node_idx', node_idx, data.y[node_idx].item())
    ax1.text(xmin, ymax - 0.1*(ymax-ymin), 'Label = {:d}'.format(data.y[node_idx].item()))
    ax1.text(xmin, ymax - 0.15*(ymax-ymin), 'Pred  = {:d}'.format(pred.argmax(dim=0).item()))

    plt.show()

    print(gt_exp.edge_imp)


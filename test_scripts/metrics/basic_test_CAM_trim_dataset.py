import random, sys
from typing import Union, List
import os
from networkx.classes.function import to_undirected
import networkx as nx
import ipdb
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from numpy import ndarray
from torch_geometric.utils import to_networkx

from graphxai.explainers import CAM, GradCAM, GNNExplainer
# from graphxai.explainers.utils.visualizations import visualize_subgraph_explanation
from graphxai.visualization.visualizations import visualize_subgraph_explanation
from graphxai.visualization.explanation_vis import visualize_node_explanation
from graphxai.gnn_models.node_classification import GCN, train, test
from graphxai.gnn_models.node_classification.testing import GCN_3layer_basic, GIN_3layer_basic

from graphxai.gnn_models.node_classification import GCN, train, test
from graphxai.gnn_models.node_classification.testing import GCN_3layer_basic, train, test

from graphxai.datasets.shape_graph import ShapeGraph
from graphxai.utils import to_networkx_conv, Explanation, distance
from graphxai.utils.perturb import rewire_edges, perturb_node_features


def graph_exp_acc(gt_exp: Explanation, generated_exp: Explanation) -> float:
    '''
    Args:
        gt_exp (Explanation): Ground truth explanation from the dataset.
        generated_exp (Explanation): Explanation output by an explainer.
    '''

    # TODO: 1) Have to implement the cases where we have multiple ground-truth explanations

    EPS = 1e-09
    thresh = 0.8
    JAC_feat = JAC_node = JAC_edge = 0    

    # Accessing the enclosing subgraph. Will be the same for both explanation.:
    exp_subgraph = generated_exp.enc_subgraph

    if generated_exp.feature_imp is not None:
        TPs = []
        FPs = []
        FNs = []
        true_feat = torch.where(gt_exp.feature_imp ==1)[0]
        for i, feat in enumerate(gt_exp.feature_imp):
            # Restore original feature numbering
            positive = generated_exp.feature_imp[i].item() > thresh
            if positive:
                if i in true_feat:
                    TPs.append(generated_exp.feature_imp[i])
                else:
                    FPs.append(generated_exp.feature_imp[i])
            else:
                if i in true_feat:
                    FNs.append(generated_exp.feature_imp[i])
        TP = len(TPs)
        FP = len(FPs)
        FN = len(FNs)
        JAC_feat = TP / (TP + FP + FN + EPS)

    if generated_exp.node_imp is not None:
        TPs = []
        FPs = []
        FNs = []
        relative_positives = (gt_exp.node_imp == 1).nonzero(as_tuple=True)[0]
        true_nodes = [gt_exp.enc_subgraph.nodes[i].item() for i in relative_positives]

        for i, node in enumerate(exp_subgraph.nodes):
            # Restore original node numbering
            positive = generated_exp.node_imp[i].item() > thresh
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
        JAC_node = TP / (TP + FP + FN + EPS)

    if generated_exp.edge_imp is not None:
        TPs = []
        FPs = []
        FNs = []
        true_edges = torch.where(gt_exp.edge_imp == 1)[0]
        for edge in range(gt_exp.edge_imp.shape[0]):
            if generated_exp.edge_imp[edge]:
                if edge in true_edges:
                    TPs.append(edge)
                else:
                    FPs.append(edge)
            else:
                if edge in true_edges:
                    FNs.append(edge)
        TP = len(TPs)
        FP = len(FPs)
        FN = len(FNs)
        JAC_edge = TP / (TP + FP + FN + EPS)
    print(JAC_feat, JAC_node, JAC_edge)
    return max(JAC_feat, JAC_node, JAC_edge)


def graph_exp_faith(generated_exp: Explanation, shape_graph: ShapeGraph, sens_idx: List[int]= [], top_k: float = 0.25) -> float:
    '''
    Args:
        gt_exp (Explanation): Ground truth explanation from the dataset.
        generated_exp (Explanation): Explanation output by an explainer.
    '''

    # TODO: 1) Implement perturbations for continuous and discrete node attribute features

    GEF_feat = GEF_node = GEF_edge = 0

    # Accessing the enclosing subgraph. Will be the same for both explanation.:
    exp_subgraph = generated_exp.enc_subgraph

    # Getting the softmax vector for the original graph
    org_vec = model(shape_graph.get_graph().x, shape_graph.get_graph().edge_index)[generated_exp.node_idx]
    org_softmax = F.softmax(org_vec, dim=-1)

    if generated_exp.feature_imp is not None:
        # Identifying the top_k features in the node attribute feature vector
        top_k_features = generated_exp.feature_imp.topk(int(generated_exp.feature_imp.shape[0] * top_k))[1]

        # node_map = [v for k, v in generated_exp.node_reference.items() if k == generated_exp.node_idx][0]

        # Getting the softmax vector for the perturbed graph
        pert_x = shape_graph.get_graph().x.clone()

        # Perturbing the unimportant node feature indices using gaussian noise
        rem_features = torch.Tensor(
            [i for i in range(shape_graph.get_graph().x.shape[1]) if i not in top_k_features]).long()

        pert_x[generated_exp.node_idx, rem_features] = perturb_node_features(x=pert_x, node_idx=generated_exp.node_idx, pert_feat=rem_features, bin_dims=sens_idx)
        # torch.normal(0, 0.1, pert_x[generated_exp.node_idx, rem_features].shape)
        pert_vec = model(pert_x, shape_graph.get_graph().edge_index)[generated_exp.node_idx]
        pert_softmax = F.softmax(pert_vec, dim=-1)
        GEF_feat = 1 - torch.exp(-F.kl_div(org_softmax.log(), pert_softmax, None, None, 'sum')).item()

    if generated_exp.node_imp is not None:
        # Identifying the top_k nodes in the explanation subgraph
        top_k_nodes = generated_exp.node_imp.topk(int(generated_exp.node_imp.shape[0] * top_k))[1]

        rem_nodes = []
        for node in range(generated_exp.node_imp.shape[0]):
            if node not in top_k_nodes:
                rem_nodes.append([k for k, v in generated_exp.node_reference.items() if v == node][0])

        # Getting the softmax vector for the perturbed graph
        pert_x = shape_graph.get_graph().x.clone()

        # Removing the unimportant nodes by masking
        pert_x[rem_nodes] = torch.zeros_like(pert_x[rem_nodes])
        pert_vec = model(pert_x, shape_graph.get_graph().edge_index)[generated_exp.node_idx]
        pert_softmax = F.softmax(pert_vec, dim=-1)
        GEF_node = 1 - torch.exp(-F.kl_div(org_softmax.log(), pert_softmax, None, None, 'sum')).item()

    if generated_exp.edge_imp is not None:
        subgraph_edges = torch.where(generated_exp.enc_subgraph.edge_mask == True)[0]
        # Get the list of all edges that we need to keep
        keep_edges = [] 
        for i in range(shape_graph.get_graph().edge_index.shape[1]):
            if i in subgraph_edges and generated_exp.edge_imp[(subgraph_edges == i).nonzero(as_tuple=True)[0]]==0:
                continue
            else:
                keep_edges.append(i)

        # Get new edge_index
        edge_index = shape_graph.get_graph().edge_index
        edge_index = edge_index[:, keep_edges]
                    
        # Getting the softmax vector for the perturbed graph
        pert_vec = model(shape_graph.get_graph().x, edge_index)[generated_exp.node_idx]
        pert_softmax = F.softmax(pert_vec, dim=-1)        
        GEF_edge = 1 - torch.exp(-F.kl_div(org_softmax.log(), pert_softmax, None, None, 'sum')).item()

    return max(GEF_feat, GEF_node, GEF_edge)


def calculate_delta(shape_graph: ShapeGraph, train_set, label, sens_idx, rep='softmax', dist_norm=2):
    delta_softmax, delta_L1, delta_L2, delta_Lfinal = [], [], [], []

    for n_id in train_set[torch.randperm(train_set.size()[0])][:100]:
        try:
            pert_edge_index = rewire_edges(shape_graph.get_graph().edge_index, node_idx=n_id.item(), num_nodes=1)
        except:
            continue
        pert_x = shape_graph.get_graph().x.clone()
        pert_x[n_id] = perturb_node_features(x=pert_x, node_idx=n_id, pert_feat=torch.arange(pert_x.shape[1]), bin_dims=sens_idx)
        # pert_x[n_id] += torch.normal(0, 0.01, pert_x[n_id].shape)
        org_vec = F.softmax(model(shape_graph.get_graph().x, shape_graph.get_graph().edge_index)[n_id], dim=-1)
        org_pred = torch.argmax(org_vec)
        pert_vec = F.softmax(model(pert_x, pert_edge_index)[n_id], dim=-1)
        pert_pred = torch.argmax(pert_vec)

        if org_pred.item() == pert_pred.item():
            if rep == 'softmax':
                # Softmax differences
                L_softmax = torch.dist(org_vec, pert_vec, p=dist_norm)
                delta_softmax.append(L_softmax.item())

            elif rep == 'intermediate':
                raise NotImplementedError('Intermediate model check will be implemented in the future version!')

            elif rep == 'final':
                raise NotImplementedError('Final embedding check will be implemented in the future version!')

            else:
                print('Invalid choice! Exiting..')

    if rep == 'softmax':
        # print(delta_softmax)
        return np.mean(delta_softmax)

    elif rep == 'intermediate':
        raise NotImplementedError('Intermediate model check will be implemented in the future version!')
        # return [np.mean(delta_L1), np.mean(delta_L2)]

    elif rep == 'final':
        raise NotImplementedError('Final embedding check will be implemented in the future version!')
        # return np.mean(delta_Lfinal)

    else:
        print('Invalid choice! Exiting...')
        exit(0)


def check_delta(shape_graph: ShapeGraph, rep, pert_x, pert_edge_index, n_id, delta, dist_norm=2):
    if rep == 'softmax':
        # Softmax differences
        org_softmax = F.softmax(model(shape_graph.get_graph().x, shape_graph.get_graph().edge_index)[n_id], dim=-1)
        org_pred = torch.argmax(org_softmax)
        pert_softmax = F.softmax(model(pert_x, pert_edge_index)[n_id], dim=-1)
        pert_pred = torch.argmax(pert_softmax)
        return torch.dist(org_softmax, pert_softmax, p=dist_norm).item() <= delta

    elif rep == 'final':
        raise NotImplementedError('Final embedding check will be implemented in the future version!')

    elif rep == 'intermediate':
        raise NotImplementedError('Intermediate model check will be implemented in the future version!')
    else:
        print('Invalid choice! Exiting..')
        exit(0)


def intersection(lst1, lst2):
    return set(lst1).union(lst2)


def graph_exp_stability(generated_exp: Explanation, shape_graph: ShapeGraph, node_id, model, delta, sens_idx, top_k=0.25, rep='softmax') -> float:
    GES = []
    num_run = 25
    for run in range(num_run):
        # Generate perturbed counterpart
        try:
            pert_edge_index = rewire_edges(shape_graph.get_graph().edge_index, node_idx=node_id.item(), num_nodes=1)  # , seed=run)
        except:
            continue
        pert_x = shape_graph.get_graph().x.clone()
        pert_x[node_id] = perturb_node_features(x=pert_x, node_idx=node_id, pert_feat=torch.arange(pert_x.shape[1]), bin_dims=sens_idx)

        if check_delta(shape_graph, rep, pert_x, pert_edge_index, node_id, delta):
            # Compute CAM explanation
            preds = model(pert_x, pert_edge_index)
            pred = preds[node_id, :].reshape(-1, 1)
            pred_class = pred.argmax(dim=0).item()
            act = lambda x: torch.argmax(x, dim=1)
            cam = CAM(model, activation=act)
            cam_pert_exp = cam.get_explanation_node(
                pert_x,
                node_idx=int(node_id),
                label=pred_class,
                edge_index=pert_edge_index)

            # Normalize the explanations to 0-1 range:
            cam_pert_exp.node_imp = cam_pert_exp.node_imp / torch.max(cam_pert_exp.node_imp)
            top_feat = int(generated_exp.node_imp.shape[0] * top_k)
            ipdb.set_trace()
            try:
                if generated_exp.node_imp.shape == cam_pert_exp.node_imp.shape:
                    ori_exp_mask = torch.zeros_like(generated_exp.node_imp)
                    ori_exp_mask[generated_exp.node_imp.topk(top_feat)[1]] = 1
                    pert_exp_mask = torch.zeros_like(cam_pert_exp.node_imp)
                    pert_exp_mask[cam_pert_exp.node_imp.topk(top_feat)[1]] = 1
                    GES.append(1 - F.cosine_similarity(ori_exp_mask.reshape(1, -1), pert_exp_mask.reshape(1, -1)).item())
                else:
                    all_nodes = [*intersection([*generated_exp.node_reference], [*cam_pert_exp.node_reference])]
                    ori_exp_mask = torch.zeros([len(all_nodes)])
                    pert_exp_mask = torch.zeros([len(all_nodes)])
                    for i, n_id in enumerate(all_nodes):
                        if n_id in [*generated_exp.node_reference]:
                            ori_exp_mask[i] = generated_exp.node_imp[generated_exp.node_reference[n_id]].item()
                        if n_id in [*cam_pert_exp.node_reference]:
                            pert_exp_mask[i] = cam_pert_exp.node_imp[cam_pert_exp.node_reference[n_id]].item()
                    topk, indices = torch.topk(ori_exp_mask, top_feat)
                    ori_exp_mask = torch.zeros_like(ori_exp_mask).scatter_(0, indices, topk)
                    ori_exp_mask[ori_exp_mask.topk(top_feat)[1]] = 1
                    topk, indices = torch.topk(pert_exp_mask, top_feat)
                    pert_exp_mask = torch.zeros_like(pert_exp_mask).scatter_(0, indices, topk)
                    pert_exp_mask[pert_exp_mask.topk(top_feat)[1]] = 1
                    GES.append(1 - F.cosine_similarity(ori_exp_mask.reshape(1, -1), pert_exp_mask.reshape(1, -1)).item())
            except:
                continue
    return max(GES)


def get_exp(explainer, node_idx, data):
    exp = explainer.get_explanation_node(
        node_idx, data.x, data.edge_index, label=data.y,
        num_hops=2, explain_feature=True)
    return exp.feature_imp, exp.edge_imp, exp.enc_subgraph.nodes, exp.enc_subgraph.edge_index, exp.enc_subgraph.edge_mask


if __name__ == '__main__':

    n = 300
    m = 1
    num_houses = 20
    train_flag = False
    bah = ShapeGraph(model_layers=3, seed=912, make_explanations=True, num_subgraphs=500, prob_connection=0.0075, subgraph_size=9, class_sep=0.5, n_informative=6, verify=True)

    # for more nodes in class 1 increase prob_connection and decrease subgraph_size

    # bah = torch.load(open('./ShapeGraph_2.pickle', 'rb'))
    # Fix the seed for reproducibility
#    np.random.seed(912)
#    torch.manual_seed(912)
#    torch.cuda.manual_seed(912)
    # ipdb.set_trace()

    data = bah.get_graph(seed=912)   
    # ipdb.set_trace()
    model = GIN_3layer_basic(64, input_feat=11, classes=2)
    # print(model)
    print('Samples in Class 0', torch.sum(data.y == 0).item())
    print('Samples in Class 1', torch.sum(data.y == 1).item())
    criterion = torch.nn.CrossEntropyLoss()

    if train_flag:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        # Train the model for 200 epochs:
        best_f1 = 0
        for epoch in range(1, 1001):
            loss = train(model, optimizer, criterion, data)
            acc, f1 = test(model, data)
            if f1 > best_f1:
                best_epoch = [loss, acc, f1]
                best_f1 = f1
                torch.save(model.state_dict(), 'model.pth')
            if epoch%50==0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Acc: {acc:.4f}, F1-score: {f1:.3f}')
       
        print(f'Best Performance: Loss: {best_epoch[0]:.4f}, Test Acc: {best_epoch[1]:.4f}, F1-score: {best_epoch[2]:.3f}')
    else:
        model.load_state_dict(torch.load('model.pth'))
        model.eval()
        test_acc, f1 = test(model, data)
        print(f'Testing Acuracy: {test_acc} | F1-score: {f1:.3f}')
        preds = model(data.x, data.edge_index)

        # Choose a node that's in class 1 (connected to 2 houses)
        class1 = (data.y == 1).nonzero(as_tuple=True)[0]
        test_idx = torch.where(data.test_mask == True)[0]
        node_idx = test_idx[torch.randperm(test_idx.shape[0])[0]]
        # ipdb.set_trace()
        # node_idx = random.choice(class1).item()
        # node_idx = 9

        pred = preds[node_idx, :].reshape(-1, 1)
        pred_class = pred.argmax(dim=0).item()

        print('GROUND TRUTH LABEL: \t {}'.format(data.y[node_idx].item()))
        print('PREDICTED LABEL   : \t {}'.format(pred.argmax(dim=0).item()))

        # if data.y[node_idx].item() == pred.argmax(dim=0).item():
        # Compute CAM explanation
        act = lambda x: torch.argmax(x, dim=1)
        cam = CAM(model, activation=act)
        cam_exp = cam.get_explanation_node(
            data.x,
            node_idx=int(node_idx),
            label=pred_class,
            edge_index=data.edge_index)

        # Explanation
        # Get ground-truth explanation
        gt_exp = bah.explanations[node_idx]

        # Compute Grad-CAM explanation
        gcam = GradCAM(model, criterion=criterion)
        # print('node_idx', node_idx)
        gcam_exp = gcam.get_explanation_node(
            data.x,
            y=data.y,
            node_idx=int(node_idx),
            edge_index=data.edge_index,
            average_variant=True)

        # Normalize the explanations to 0-1 range:
        cam_exp.node_imp = cam_exp.node_imp / torch.max(cam_exp.node_imp)
        gcam_exp.node_imp = gcam_exp.node_imp / torch.max(gcam_exp.node_imp)

	# Compute Score
        cam_gea_score = graph_exp_acc(gt_exp, cam_exp)
        cam_gef_score = graph_exp_faith(cam_exp, bah, sens_idx=[bah.sensitive_feature])
        delta = calculate_delta(bah, torch.where(data.train_mask == True)[0], label=data.y, sens_idx=[bah.sensitive_feature])
        # cam_ges_score = graph_exp_stability(cam_exp, bah, node_id=node_idx, model=model, delta=delta, sens_idx=[bah.sensitive_feature])

        print('### CAM ###')
        print(f'Graph Explanation Accuracy using CAM={cam_gea_score:.3f}')
        print(f'Graph Explanation Faithfulness using CAM={cam_gef_score:.3f}')
        # print(f'Graph Explanation Stability using CAM={cam_ges_score:.3f}')
        print(f'Delta: {delta:.3f}')
     
        # Test for GNN Explainers
        gnnexpr = GNNExplainer(model)
        pred_exp = gnnexpr.get_explanation_node(x=data.x, node_idx=int(node_idx), edge_index=data.edge_index)
        gnnex_gea_score = graph_exp_acc(gt_exp, pred_exp)
        gnnex_gef_score = graph_exp_faith(pred_exp, bah, sens_idx=[bah.sensitive_feature])
        gnnex_ges_score = graph_exp_stability(pred_exp, bah, node_id=node_idx, model=model, delta=delta, sens_idx=[bah.sensitive_feature])
        print('### GNNExplainer ###')
        print(f'Graph Explanation Accuracy using GNNExplainer={gnnex_gea_score:.3f}')
        print(f'Graph Explanation Stability using GNNExplainer={gnnex_ges_score:.3f}')

        # gcam_gea_score = graph_exp_acc(gt_exp, gcam_exp)
        # gcam_gef_score = graph_exp_faith(gcam_exp)
        # gcam_ges_score = graph_exp_stability(gcam_exp, node_id=node_idx, model=model, delta=delta)
        # print('\n### GradCAM ###')
        # print(f'Graph Explanation Accuracy using GradCAM={gcam_gea_score:.3f}')
        # print(f'Graph Explanation Faithfulness using GradCAM={gcam_gef_score:.3f}')
        # print(f'Graph Explanation Stability using GradCAM={gcam_ges_score:.3f}')

        # # Plotting:
        # fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        #
        # gt_exp.context_draw(num_hops=bah.num_hops, graph_data=data, additional_hops=1, heat_by_exp=True, ax=ax1)
        # ax1.set_title('Ground Truth')
        #
        # cam_exp.context_draw(num_hops=bah.num_hops, graph_data=data, additional_hops=1, heat_by_exp=True, ax=ax2)
        #
        # ax2.set_title('CAM')
        # ymin, ymax = ax2.get_ylim()
        # xmin, xmax = ax2.get_xlim()
        # ax2.text(xmin, ymin, 'Faithfulness: {:.3f}'.format(cam_gea_score))
        # gcam_exp.context_draw(num_hops=bah.num_hops, graph_data=data, additional_hops=1, heat_by_exp=True, ax=ax3)
        # ax3.set_title('Grad-CAM')
        # ymin, ymax = ax3.get_ylim()
        # xmin, xmax = ax3.get_xlim()
        # ax3.text(xmin, ymin, 'Faithfulness: {:.3f}'.format(gcam_gea_score))
        #
        # ymin, ymax = ax1.get_ylim()
        # xmin, xmax = ax1.get_xlim()
        # # print('node_idx', node_idx, data.y[node_idx].item())
        # ax1.text(xmin, ymax - 0.1 * (ymax - ymin), 'Label = {:d}'.format(data.y[node_idx].item()))
        # ax1.text(xmin, ymax - 0.15 * (ymax - ymin), 'Pred  = {:d}'.format(pred.argmax(dim=0).item()))
        #
        # plt.savefig('demo.pdf', bbox_inches='tight')




                # # Layer 1 differences
                # org_layer1 = F.relu(model.gin1(shape_graph.get_graph().x, shape_graph.get_graph().edge_index))
                # pert_layer1 = F.relu(model.gin1(pert_x, pert_edge_index))
                # L1 = torch.dist(org_layer1[n_id], pert_layer1[n_id], p=dist_norm)
                #
                # # Layer 2 differences
                # org_layer2 = F.relu(model.gin2(org_layer1, generated_exp.graph.edge_index))
                # pert_layer2 = F.relu(model.gin2(pert_layer1, pert_edge_index))
                # L2 = torch.dist(org_layer2[n_id], pert_layer2[n_id], p=dist_norm)
                # delta_L1.append(L1.item())
                # delta_L2.append(L2.item())

                # # Layer 3 differences
                # org_layer3 = model(shape_graph.get_graph().x, shape_graph.get_graph().edge_index)
                # # model.gin3(org_layer2, generated_exp.graph.edge_index)
                # pert_layer3 = model(pert_x, pert_edge_index)
                # # model.gin3(pert_layer2, pert_edge_index)
                # Lfinal = torch.dist(org_layer3[n_id], pert_layer3[n_id], p=dist_norm)
                # delta_Lfinal.append(Lfinal.item())

        # Final embedding differences
        # org_layer3 = model(shape_graph.get_graph().x, shape_graph.get_graph().edge_index)
        # pert_layer3 = model(pert_x, pert_edge_index)
        # return torch.dist(org_layer3[n_id], pert_layer3[n_id], p=dist_norm).item() <= delta


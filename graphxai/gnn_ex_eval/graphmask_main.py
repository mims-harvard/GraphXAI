import math
import tqdm
import copy

import numpy as np

import torch
import torch.nn.functional as F

import warnings
warnings.filterwarnings('ignore')

from utils import *
from models import *
from explainers import *
from metrics_edge import EdgeMetrics
from torch_geometric.data import Data
from load_model_opt import ModelDataLoader
from torch_geometric.utils import k_hop_subgraph
from networkx.algorithms.swap import double_edge_swap as swap
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import convert
from networkx.linalg.graphmatrix import adjacency_matrix as adj_mat

from GraphMask.utils.moving_average import MovingAverage
from GraphMask.utils.lagrangian_optimization import LagrangianOptimization
from GraphMask.utils.hard_concrete import HardConcrete
from GraphMask.utils.multiple_inputs_layernorm_linear import MultipleInputsLayernormLinear
from GraphMask.utils.squeezer import Squeezer
from torch_sparse import SparseTensor, matmul
from typing import Union, Tuple
from torch import Tensor
from torch_geometric.typing import OptPairTensor, Adj, Size


class SAGEConvLayer(MessagePassing):
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = True,
                 bias: bool = True, graphmaskmode: bool =False, gate=None, baseline=None, **kwargs):  # yapf: disable
        kwargs.setdefault('aggr', 'mean')
        super(SAGEConvLayer, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = torch.nn.Linear(in_channels[0], out_channels, bias=bias)
        self.lin_r = torch.nn.Linear(in_channels[1], out_channels, bias=False)

        self.graphmaskmode = graphmaskmode
        if gate is not None:
            self.gate = gate
            self.baseline = baseline

        self.gate_storage = {}

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, graphmaskmode=False,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, graphmaskmode=graphmaskmode)
        out = self.lin_l(out)

        x_r = x[1]
        if x_r is not None:
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_i, x_j, graphmaskmode):
        if graphmaskmode:
            gate, penalty = self.gate([x_i, x_j])
            self.penalty += len(x_i) / self.num_of_edges * penalty
            self.num_masked += len(torch.where(gate.reshape(-1) != 1)[0])

            message = gate.unsqueeze(-1) * x_i + (1 - gate.unsqueeze(-1)) * self.baseline.unsqueeze(0)

            if self.return_gates:
                self.gate_storage[0] = copy.deepcopy(gate.to('cpu').detach())
        else:
            message = x_j
        return message

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def graphmask_forward(self, x, edge_index, graphmask_mode, return_gates):
        self.return_gates = return_gates
        self.penalty = 0
        self.num_masked = 0
        self.num_of_edges = edge_index.shape[1]

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        gm_out = self.propagate(edge_index, x=x, graphmaskmode=graphmask_mode)
        gm_out = self.lin_l(gm_out)

        x_r = x[1]
        if x_r is not None:
            gm_out += self.lin_r(x_r)

        if self.normalize:
            gm_out = F.normalize(gm_out, p=2., dim=-1)

        return gm_out, self.penalty, self.num_masked

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_size, hidden_size, out_size, args, device, graphmask_mode=False):
        super(GraphSAGE, self).__init__()
        # create layers
        self.args = args
        self.device = device

        if graphmask_mode:
            gates = []
            baselines = []

            # For GCNLayer
            # vertex_embedding_dims = [hidden_size, hidden_size]
            # message_dims = [hidden_size, out_size]
            # h_dims = message_dims

            # For SAGELayer
            vertex_embedding_dims = [hidden_size, hidden_size]
            message_dims = [in_size, out_size]
            h_dims = message_dims

            for v_dim, m_dim, h_dim in zip(vertex_embedding_dims, message_dims, h_dims):
                # For GCNLayer
                # gate_input_shape = [m_dim, m_dim]

                # For SAGEConvLayer
                gate_input_shape = [m_dim, m_dim]

                # this gate is the core:
                # hard concrete layer is to reparametrize the L0 norm..
                # hardconcrete layer outputs a

                # different layers have different gates
                gate = torch.nn.Sequential(
                    MultipleInputsLayernormLinear(gate_input_shape, 32),
                    torch.nn.ReLU(),
                    torch.nn.Linear(32, 1),
                    Squeezer(),
                    HardConcrete()
                )

                gates.append(gate)

                baseline = torch.FloatTensor(m_dim)
                stdv = 1. / math.sqrt(m_dim)
                baseline.uniform_(-stdv, stdv)
                baseline = torch.nn.Parameter(baseline, requires_grad=True)

                baselines.append(baseline)

            gates = torch.nn.ModuleList(gates)
            self.gates = gates

            baselines = torch.nn.ParameterList(baselines)
            self.baselines = baselines

            # Initially we cannot update any parameters. They should be enabled layerwise
            for parameter in self.parameters():
                parameter.requires_grad = False

            self.conv1 = SAGEConvLayer(in_size, hidden_size, graphmaskmode=graphmask_mode, gate=self.gates[0],
                                        baseline=self.baselines[0])
            self.conv2 = SAGEConvLayer(hidden_size, out_size, graphmaskmode=graphmask_mode, gate=self.gates[1],
                                        baseline=self.baselines[1])
        else:
            self.conv1 = SAGEConvLayer(in_size, hidden_size)
            self.conv2 = SAGEConvLayer(hidden_size, out_size)

        self.fc = torch.nn.Linear(out_size, 2)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def enable_layer(self, layer):
        print("Enabling layer " + str(layer))

        for parameter in self.gates[layer].parameters():
            parameter.requires_grad = True

        self.baselines[layer].requires_grad = True

    def graphmask_forward(self, x, edge_index, graphmask_mode=False, return_gates=False):
        out, penalty_l1, num_masked_l1 = self.conv1.graphmask_forward(x, edge_index, graphmask_mode, return_gates)
        out = F.relu(out)
        out, penalty_l2, num_masked_l2 = self.conv2.graphmask_forward(out, edge_index, graphmask_mode, return_gates)
        out = self.fc(out)
        return F.softmax(out, dim=-1), penalty_l1 + penalty_l2, [num_masked_l1, num_masked_l2]

    def forward(self, feat, edge_index):
        h_dict = self.conv1(feat, edge_index)
        h_dict = F.relu(h_dict)
        h = self.conv2(h_dict, edge_index)
        out = self.fc(h)
        return F.softmax(out, dim=-1)

    def get_vertex_embedding_dims(self):
        return np.array([self.conv1.in_size, self.conv2.in_size])

    def get_message_dims(self):
        return np.array([self.conv1.out_size, self.conv2.out_size])

    def count_layers(self):
        return 2

    def get_gates(self):
        return [self.conv1.gate_storage, self.conv2.gate_storage]


def rewire_edges(x, edge_index, degree):
    # Convert to networkx graph for rewiring edges
    data = Data(x=x, edge_index=edge_index)
    G = convert.to_networkx(data, to_undirected=True)
    rewired_G = swap(G, nswap=degree, max_tries=degree * 25, seed=912)
    rewired_adj_mat = adj_mat(rewired_G)
    rewired_edge_indexes = convert.from_scipy_sparse_matrix(rewired_adj_mat)[0]
    return rewired_edge_indexes


def calculate_edge_exp_metrics(node_idx, subset, sub_edge_index, mapping, sub_x, metrics, model, device,
                               set_categorical_mask, sens_idx, degree):
    # Single node
    sub_x.requires_grad = True
    model = model.to(device)
    ind = torch.where(subset == node_idx)[0].item()

    # Local-Group Faithfulness
    perturbed_nodes = [sub_x.clone()]
    upd_out, penalty, num_masked = model.graphmask_forward(sub_x.to(device), sub_edge_index.to(device),
                                                           graphmask_mode=True, return_gates=True)

    node_explanation = model.get_gates()[0][0]

    top_k_features = 0.25
    top_k = np.ceil(top_k_features * sub_edge_index.shape[1])

    # Stability
    with torch.no_grad():
        cont_noise = torch.ones(sub_x.shape[1]).normal_(0, args.var).mul(1 - set_categorical_mask[:, 1])
        catg_temp = torch.zeros_like(cont_noise)
        pert_sub_x = sub_x.clone()
        for ii in range(catg_temp.shape[0]):
            # categorical flip with some prob p
            if set_categorical_mask[ii][1] == 1 and torch.rand(1) <= 0.1:
                catg_temp[ii] = sub_x[ind, ii]
                pert_sub_x[ind, ii] = torch.randint(torch.unique(sub_x[:, ii].detach())[0].int().item(),
                                                    torch.unique(sub_x[:, ii].detach())[-1].int().item() + 1, (1,))

        pert_sub_x[ind, :] += cont_noise
    pert_sub_x.requires_grad = True

    try:
        rewire_edge_index = rewire_edges(x=sub_x, edge_index=sub_edge_index, degree=degree)[:, :sub_edge_index.shape[1]]
    except:
        return None

    # get predictions for original and perturbed node
    out_x = model(sub_x.to(device), sub_edge_index.to(device))
    preds_x = torch.argmax(out_x, dim=-1)

    out_pert_x = model(pert_sub_x.to(device), rewire_edge_index.to(device))
    preds_pert_x = torch.argmax(out_pert_x, dim=-1)

    if preds_x[mapping].item() == preds_pert_x[mapping].item():
        model = model.to(device)
        _, _, _ = model.graphmask_forward(pert_sub_x.to(device), rewire_edge_index.to(device),
                                          graphmask_mode=True, return_gates=True)

        pert_node_explanation = model.get_gates()[0][0]
        stab_score = metrics.dist_explanation(node_explanation, pert_node_explanation, top_k, ind)
    else:
        stab_score = None
        pert_node_explanation = torch.zeros_like(node_explanation)

    # Counterfactual Fairness
    with torch.no_grad():
        counter_sub_x = sub_x.clone()
        counter_sub_x[ind, sens_idx] = 1 - counter_sub_x[ind, sens_idx]
    counter_sub_x.requires_grad = True

    model = model.to(device)
    _, _, _ = model.graphmask_forward(counter_sub_x.to(device), sub_edge_index.to(device),
                                      graphmask_mode=True, return_gates=True)

    counter_node_explanation = model.get_gates()[0][0]

    # get predictions for original and perturbed node
    out_x = model(sub_x.to(device), sub_edge_index.to(device))
    preds_x = torch.argmax(out_x, dim=-1)

    out_counter_x = model(counter_sub_x.to(device), sub_edge_index.to(device))
    preds_counter_x = torch.argmax(out_counter_x, dim=-1)

    if preds_x[mapping].item() == preds_counter_x[mapping].item():
        counter_fair_score = metrics.dist_explanation(node_explanation, counter_node_explanation, top_k, ind)
    else:
        counter_fair_score = None

    for _ in range(args.num_samples):
        cont_noise = torch.ones(sub_x.shape[1]).normal_(0, args.var).mul(1 - set_categorical_mask[:, 1])
        catg_temp = torch.zeros_like(cont_noise)

        for ii in range(catg_temp.shape[0]):
            # categorical flip with some prob p  and torch.rand(1) <= 0.1:
            if set_categorical_mask[ii][1] == 1 and torch.rand(1) <= 0.1:
                catg_temp[ii] = sub_x[ind, ii]
                sub_x[ind, ii] = torch.randint(torch.unique(sub_x[:, ii].detach())[0].int().item(),
                                               torch.unique(sub_x[:, ii].detach())[-1].int().item() + 1, (1,))
        sub_x[ind, :] += cont_noise
        perturbed_nodes.append(sub_x.clone())
        sub_x[ind, :] -= cont_noise

        for ii in range(catg_temp.shape[0]):
            if set_categorical_mask[ii][1] == 1:
                sub_x[ind, ii] = catg_temp[ii]

    # Local-Group Faithfulness
    lg_pred_score, x_norm, lg_softmax_score, lg_norm_softmax_score = metrics.local_group_faithfulness(perturbed_nodes,
                                                                                                      node_explanation,
                                                                                                      args.num_samples,
                                                                                                      top_k, ind,
                                                                                                      degree)

    # Local-Group Fairness
    lg_fair_score, fairness_bound = metrics.local_group_fairness(perturbed_nodes, subset, node_explanation,
                                                                 args.num_samples, top_k, ind,
                                                                 sens_idx, degree)

    return lg_pred_score, stab_score, counter_fair_score, lg_fair_score, x_norm, lg_softmax_score, \
           lg_norm_softmax_score, torch.cat(
        [node_explanation.unsqueeze(0), pert_node_explanation.unsqueeze(0), counter_node_explanation.unsqueeze(0)],
        dim=0), fairness_bound


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=912, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='bail',
                    choices=['bail', 'credit', 'german'])
parser.add_argument("--num_heads", type=int, default=1,
                        help="number of hidden attention heads")
parser.add_argument("--num_out_heads", type=int, default=1,
                    help="number of output attention heads")
parser.add_argument("--num_layers", type=int, default=2,
                    help="number of hidden layers")
parser.add_argument('--allowance', default=0.005, type=int, help='allowance')
parser.add_argument('--moving_average_window_size', default=100, type=int, help='moving_average_window_size')
parser.add_argument('--penalty_scaling', default=1, type=int, help='penalty_scaling')
parser.add_argument('--model', type=str, default='gcn',
                    choices=['gcn', 'gat', 'sage', 'gin', 'jk', 'infomax', 'ssf', 'rogcn', 'jaccard'])
parser.add_argument('--var', type=float, default=1, help='standard deviation of the noise')
parser.add_argument('--num_samples', type=int, default=50, help='number of samples in the local-group metrics')

args = parser.parse_known_args()[0]
args.cuda = not args.no_cuda and torch.cuda.is_available()

# set seeds
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:  # torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

# torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def main():
    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # %%
    model_data_loader = ModelDataLoader(args)

    # Load data
    adj, features, labels, idx_train, idx_val, idx_test, _, sens_idx, set_categ_mask = model_data_loader.load_dataset()
    edge_index = convert.from_scipy_sparse_matrix(adj)[0]

    # # Train the model first
    # # Model and optimizer
    # model = GCN(in_size=features.shape[1],
    #             hidden_size=args.hidden,
    #             args=args,
    #             out_size=16,
    #             device=device,
    #             graphmask_mode=False).to(device)
    #
    # # Train the model first
    # best_loss = 100
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # for epoch in range(1000):
    #     model.train()
    #     optimizer.zero_grad()
    #     output = model(features.to(device), edge_index.to(device))
    #
    #     # NLL_Loss for NeurIPS paper as GLIME needs softmax outputs
    #     loss_train = F.nll_loss(output[idx_train], labels[idx_train].to(device))
    #     auc_roc_train = roc_auc_score(labels.cpu().numpy()[idx_train], output.detach().cpu().numpy()[idx_train][:, 1])
    #     loss_train.backward()
    #     optimizer.step()
    #
    #     # Evaluate validation set performance separately,
    #     model.eval()
    #     output = model(features.to(device), edge_index.to(device))
    #
    #     # NLL_Loss for NeurIPS paper as GLIME needs softmax outputs
    #     loss_val = F.nll_loss(output[idx_val], labels[idx_val].to(device))
    #     auc_roc_val = roc_auc_score(labels.cpu().numpy()[idx_val], output.detach().cpu().numpy()[idx_val][:, 1])
    #
    #     # if epoch % 100 == 0:
    #     #     print(f"[Train] Epoch {epoch}:train_loss: {loss_train.item():.4f} | train_auc_roc: {auc_roc_train:.4f} | val_loss: {loss_val.item():.4f} | val_auc_roc: {auc_roc_val:.4f}")
    #     if loss_val.item() < best_loss:  # acc_val > best_acc:
    #         best_loss = loss_val.item()
    #         torch.save(model.state_dict(), f'weights_graphmask_{args.dataset}.pt')
    # model.load_state_dict(torch.load(f'weights_graphmask_{args.dataset}.pt'))
    # model.eval()
    # output = model(features.to(device), edge_index.to(device))
    # output_preds = torch.argmax(output, dim=-1)
    # auc_roc_test = roc_auc_score(labels.cpu().numpy()[idx_test.cpu()],
    #                              output.detach().cpu().numpy()[idx_test.cpu()][:, 1])
    # f1_s = f1_score(labels[idx_test].cpu().numpy(), output_preds[idx_test].cpu().numpy())
    # print("The AUCROC of pre-trained model: {:.4f}".format(auc_roc_test))
    # print(f'F1-score: {f1_s}')
    # exit(0)

    # Load pre-trained weights for post-hoc explanations
    model = GraphSAGE(in_size=features.shape[1],
                hidden_size=args.hidden,
                args=args,
                out_size=args.hidden,
                device=device,
                graphmask_mode=True).to(device)

    model_dict = model.state_dict()
    pretrained_dict = torch.load(f'weights_sage_{args.dataset}.pt')
    for k, v in pretrained_dict.items():
        if k in model_dict:
            model_dict[k] = pretrained_dict[k]
    # pretrained_dict = {k for k, v in pretrained_dict.items() if k in model_dict}
    # model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    # disable all gradient
    def disable_all_gradients(module):
        for param in module.parameters():
            param.requires_grad = False
    disable_all_gradients(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    lagrangian_optimization = LagrangianOptimization(optimizer,
                                                     device,
                                                     batch_size_multiplier=1)

    f_moving_average = MovingAverage(window_size=args.moving_average_window_size)
    g_moving_average = MovingAverage(window_size=args.moving_average_window_size)
    loss_fct = torch.nn.MSELoss()

    # iterate over layers. One at a time!
    for layer in reversed(list(range(model.count_layers()))):
        model.enable_layer(layer)  # enable baselines and gates parameters
        if args.dataset == 'german':
            num_epoch = 90
        else:
            num_epoch = 101
        for epoch in range(num_epoch):
            model.train()
            org_out, _, _ = model.graphmask_forward(features.to(device), edge_index.to(device), graphmask_mode=False)
            org_preds = torch.argmax(org_out, dim=-1)
            loss_org_pred = F.nll_loss(org_out[idx_train], labels[idx_train].to(device))

            upd_out, penalty, num_masked = model.graphmask_forward(features.to(device), edge_index.to(device),
                                                                   graphmask_mode=True)
            upd_preds = torch.argmax(org_out, dim=-1)
            loss_upd_pred = F.nll_loss(upd_out[idx_train], labels[idx_train].to(device))

            loss = loss_fct(org_preds[idx_train].float(), upd_preds[idx_train].float())

            g = torch.relu(loss - args.allowance).mean()
            f = penalty * args.penalty_scaling

            lagrangian_optimization.update(f, g)

            f_moving_average.register(float(f.item()))
            g_moving_average.register(float(loss.mean().item()))
            # if epoch % 50 == 0:
            #     print(
            #         "Running epoch {0:n} of GraphMask training. Mean divergence={1:.4f}, mean penalty={2:.4f}, bce_update={3:.4f}, bce_original={4:.4f}, num_masked_l1={5:.4f}, num_masked_l2={6:.4f}".format(
            #             epoch,
            #             g_moving_average.get_value(),
            #             f_moving_average.get_value(),
            #             loss_upd_pred,
            #             loss_org_pred,
            #             num_masked[0] / edge_index.shape[1],
            #             num_masked[1] / edge_index.shape[1])
            #     )

    # set degree for rewiring
    if args.dataset == 'german':
        degree = 45
    elif args.dataset == 'bail':
        degree = 34
    else:
        degree = 96

    lg_norm_sm_faithfulness = []
    stability = []
    counter_fair = []
    lg_fairness = []
    count = 0
    model.eval()
    for idx in tqdm.tqdm(np.random.randint(0, idx_test.shape[0], idx_test.shape[0])):
        node_idx = idx_test[idx].item()
        num_hops = 1
        subset, sub_edge_index, mapping, hard_edge_mask = k_hop_subgraph(
            node_idx, num_hops, edge_index, relabel_nodes=True,
            num_nodes=features.size(0))
        sub_x = features[subset]

        exp_edge_metrics = EdgeMetrics(model, labels.to(device), sub_edge_index, mapping, 'graphmask', device)
        score = calculate_edge_exp_metrics(node_idx, subset, sub_edge_index, mapping, sub_x, exp_edge_metrics,
                                           model, device, set_categ_mask, sens_idx, degree)

        # Add scores
        if score is not None:
            if score[1] is not None and score[2] is not None and score[3] == score[3] and score[4] == score[4]:
                lg_norm_sm_faithfulness.append(score[6])
                lg_fairness.append(score[3])
                stability.append(score[1].item())
                counter_fair.append(score[2].item())
                count += 1

    print(
        f'Faithfulness: {np.array(lg_norm_sm_faithfulness).mean():.4f}+-{np.array(lg_norm_sm_faithfulness).std() / np.sqrt(count):.4f}')
    print(f'Stability: {np.array(stability).mean():.4f}+-{np.array(stability).std()/np.sqrt(count):.4f}')
    print(f'Counterfactual Fairnss: {np.array(counter_fair).mean():.4f}+-{np.array(counter_fair).std()/np.sqrt(count):.4f}')
    print(f'Fairness: {np.array(lg_fairness).mean():.4f}+-{np.array(lg_fairness).std()/np.sqrt(count):.4f}')


if __name__ == "__main__":
    main()

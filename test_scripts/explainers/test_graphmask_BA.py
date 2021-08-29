import math
import copy

import numpy as np

import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

from torch_sparse import SparseTensor, matmul
from typing import Union, Tuple
from torch import Tensor
from torch_geometric.typing import OptPairTensor, Adj, Size

from graphxai.gnn_models.node_classification import BA_Houses, train, test
from graphxai.explainers.graphmask import LagrangianOptimization, HardConcrete, \
    MultipleInputsLayernormLinear, Squeezer


class SAGEConvLayer(MessagePassing):
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = True, bias: bool = True,
                 gate: torch.nn.Module = None, baseline: torch.nn.Module = None,
                 **kwargs):
        kwargs.setdefault('aggr', 'mean')
        super(SAGEConvLayer, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = torch.nn.Linear(in_channels[0], out_channels, bias=bias)
        self.lin_r = torch.nn.Linear(in_channels[1], out_channels, bias=False)

        if gate is not None:
            self.gate = gate
            self.baseline = baseline

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

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, graphmask_mode=False,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, graphmask_mode=graphmask_mode)
        out = self.lin_l(out)

        x_r = x[1]
        if x_r is not None:
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_i, x_j, graphmask_mode):
        if graphmask_mode:
            gate, penalty = self.gate([x_i, x_j])
            self.penalty += len(x_i) / self.num_of_edges * penalty
            self.num_masked += len(torch.where(gate.reshape(-1) != 1)[0])

            message = gate.unsqueeze(-1) * x_i + (1 - gate.unsqueeze(-1)) * self.baseline.unsqueeze(0)

            if self.return_gates:
                self.gate_storage = copy.deepcopy(gate.detach())
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
        gm_out = self.propagate(edge_index, x=x, graphmask_mode=graphmask_mode)
        gm_out = self.lin_l(gm_out)

        x_r = x[1]
        if x_r is not None:
            gm_out += self.lin_r(x_r)

        if self.normalize:
            gm_out = F.normalize(gm_out, p=2., dim=-1)

        return gm_out, self.penalty, self.num_masked

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__,
                                   self.in_channels, self.out_channels)


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_size, hidden_size, out_size, graphmask_mode=False):
        super(GraphSAGE, self).__init__()
        # create layers
        if graphmask_mode:
            gates = []
            baselines = []

            vertex_embedding_dims = [hidden_size, hidden_size]
            message_dims = [in_size, out_size]
            h_dims = message_dims

            for v_dim, m_dim, h_dim in zip(vertex_embedding_dims, message_dims, h_dims):
                gate_input_shape = [m_dim, m_dim]

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

            # Initially we cannot update any parameters. They should be enabled layerwise.
            for parameter in self.parameters():
                parameter.requires_grad = False

            self.conv1 = SAGEConvLayer(in_size, hidden_size,
                                       gate=self.gates[0], baseline=self.baselines[0])
            self.conv2 = SAGEConvLayer(hidden_size, out_size,
                                       gate=self.gates[1], baseline=self.baselines[1])
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


n = 300
m = 2
num_houses = 30

bah = BA_Houses(n, m)
data, inhouse = bah.get_data(num_houses, multiple_features=True)

model = GraphSAGE(in_size=data.x.shape[1], hidden_size=16, out_size=16,
                  graphmask_mode=True)

print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(1, 201):
    loss = train(model, optimizer, criterion, data)
    acc = test(model, data)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Acc: {acc:.4f}')

# Train graphmask model
for param in model.parameters():
    param.requires_grad = False

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
lagrangian_optimization = LagrangianOptimization(optimizer, batch_size_multiplier=1)

allowance = 0.1
penalty_scaling = 0.1
loss_fn = torch.nn.MSELoss()

for layer in reversed(list(range(model.count_layers()))):
    model.enable_layer(layer)  # Enable baselines and gates parameters
    for epoch in range(1, 201):
        model.train()
        org_out, _, _ = model.graphmask_forward(data.x, data.edge_index,
                                                graphmask_mode=False)
        loss_org_pred = criterion(org_out[data.train_mask], data.y[data.train_mask])
        org_preds = torch.argmax(org_out, dim=-1)

        upd_out, penalty, num_masked = \
            model.graphmask_forward(data.x, data.edge_index,
                                    graphmask_mode=True, return_gates=True)

        loss_upd_pred = criterion(upd_out[data.train_mask], data.y[data.train_mask])
        upd_preds = torch.argmax(org_out, dim=-1)

        loss = loss_fn(org_preds[data.train_mask].float(),
                       upd_preds[data.train_mask].float())

        g = torch.relu(loss - allowance).mean()
        f = penalty * penalty_scaling

        lagrangian_optimization.update(f, g)

exp_by_layer = model.get_gates()

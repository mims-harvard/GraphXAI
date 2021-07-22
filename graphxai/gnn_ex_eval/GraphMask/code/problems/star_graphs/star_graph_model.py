import torch
from torch.nn import Linear, LayerNorm
from code.gnns.rgcn import RGCN
from code.abstract.abstract_torch_module import AbstractTorchModule
from code.utils.evaluation.binary_classification_model_output import BinaryClassificationModelOutput
import numpy as np


class StarGraphModel(AbstractTorchModule):

    n_classes = None
    gcn_dim = None
    device = None

    def __init__(self, configuration):
        AbstractTorchModule.__init__(self)
        self.n_classes = configuration["task"]["n_colours"]
        self.gcn_dim = configuration["task"]["gcn_dim"]

        self.initial_transform = torch.nn.Sequential(
            Linear(2 * self.n_classes, self.gcn_dim),
            LayerNorm(self.gcn_dim),
            torch.nn.ReLU(),
            Linear(self.gcn_dim, self.gcn_dim),
            LayerNorm(self.gcn_dim),
            torch.nn.ReLU(),
        )

        self.final_transform = torch.nn.Sequential(
            Linear(self.gcn_dim, self.gcn_dim * 2),
            LayerNorm(self.gcn_dim * 2),
            torch.nn.ReLU(),
            Linear(self.gcn_dim * 2, self.gcn_dim),
            LayerNorm(self.gcn_dim),
            torch.nn.ReLU(),
            Linear(self.gcn_dim, 1))

        self.loss = torch.nn.BCEWithLogitsLoss(reduction="none")

        self.gnn = RGCN(self.gcn_dim,
                        self.gcn_dim,
                        n_relations=self.n_classes,
                        n_layers=configuration["model_parameters"]["gnn_layers"],
                        inverse_edges=False)

    def set_device(self, device):
        self.to(device)
        self.device = device

        if self.gnn is not None:
            self.gnn.set_device(device)

    def forward(self, batch):
        vertex_cumulative_count = []
        carryover = 0
        for example in batch:
            if len(vertex_cumulative_count) == 0:
                vertex_cumulative_count.append(0)
            else:
                vertex_cumulative_count.append(vertex_cumulative_count[-1])
                vertex_cumulative_count[-1] += carryover

            carryover = self.retrieve_vertex_count(example)

        target_vertex_locations = torch.LongTensor(vertex_cumulative_count).to(self.device)
        joint_edge_types = torch.cat([torch.LongTensor(self.retrieve_edge_types(example)).to(self.device) for example in batch], 0)

        joint_edges = []
        for i, example in enumerate(batch):
            edge_list = torch.LongTensor(self.retrieve_edges(example)).to(self.device) + vertex_cumulative_count[i]
            joint_edges.append(edge_list)

        joint_edges = torch.cat(joint_edges, 0)
        joint_edges = joint_edges.transpose(1, 0)

        joint_vertex_input = torch.cat([torch.Tensor(self.retrieve_vertex_input(example)).to(self.device) for example in batch], 0)
        joint_vertex_labels = torch.LongTensor([self.retrieve_label(example) for example in batch]).to(self.device)

        vertex_embeddings = self.initial_transform(joint_vertex_input)
        vertex_embeddings = self.gnn(vertex_embeddings=vertex_embeddings,
                                     edges=joint_edges,
                                     edge_types=joint_edge_types)

        target_vertices = torch.index_select(vertex_embeddings, 0, target_vertex_locations)

        scores = self.final_transform(target_vertices).squeeze(-1)
        loss = self.loss(scores, joint_vertex_labels.float())
        predictions = torch.sigmoid(scores)

        predictions_by_example = []
        for i, example in enumerate(batch):
            label = self.retrieve_label(example)
            prediction = predictions[i].detach().cpu().numpy()

            example_output = BinaryClassificationModelOutput(np.array([prediction]), np.array([label]))
            predictions_by_example.append(example_output)

        return loss, predictions_by_example

    def retrieve_vertex_count(self, example):
        return len(example[0])

    def retrieve_edge_types(self, example):
        return example[2]

    def retrieve_edges(self, example):
        return example[1]

    def retrieve_vertex_input(self, example):
        return example[0]

    def retrieve_label(self, example):
        return example[3]

    def count_messages(self, batch):
        edge_count = sum([len(self.retrieve_edges(example)) for example in batch])
        return edge_count

    def get_gnn(self):
        return self.gnn

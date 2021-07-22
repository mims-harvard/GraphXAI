import scipy
from torch.nn import Dropout, CrossEntropyLoss

from code.abstract.abstract_torch_module import AbstractTorchModule
import torch
import numpy as np

from code.gnns.qa_gnn import QaGNN
from code.utils.evaluation.choice_model_output import ChoiceModelOutput
from code.utils.torch_utils.xavier_linear import XavierLinear


class QAModel(AbstractTorchModule):
    n_edge_types = 4

    def __init__(self, configuration):
        AbstractTorchModule.__init__(self)

        self.layers = configuration["model_parameters"]["gnn_layers"]
        self.configuration = configuration
        self.max_nodes = configuration["task"]["max_nodes"]
        self.max_query_size = configuration["task"]["max_query_size"]
        self.max_candidates = configuration["task"]["max_candidates"]

        embedding_input_dim = 300

        self.gcn = QaGNN(dim=512,
                         n_layers=self.layers,
                         n_relations=self.n_edge_types,
                         share_parameters=True)

        self.node_compress_mlp = torch.nn.Sequential(XavierLinear(embedding_input_dim, 256),
                                                     torch.nn.Tanh(),
                                                     torch.nn.Dropout(p=0.2))

        self.node_mlp = torch.nn.Sequential(XavierLinear(512, 1024),
                                            torch.nn.Tanh(),
                                            torch.nn.Dropout(p=0.2),
                                            XavierLinear(1024, 512),
                                            torch.nn.Tanh(),
                                            torch.nn.Dropout(p=0.2))

        # self.lstm = LSTM(3072, 256, 2, batch_first=True, bidirectional=True)

        self.lstm1 = torch.nn.LSTM(embedding_input_dim, 256, num_layers=1, batch_first=True, bidirectional=True,
                                   dropout=0)
        self.lstm2 = torch.nn.LSTM(512, 128, num_layers=1, batch_first=True, bidirectional=True, dropout=0)
        self.query_dropout = Dropout(p=0.2)

        self.second_mlp = torch.nn.Sequential(XavierLinear(768, 128),
                                              torch.nn.Tanh(),
                                              XavierLinear(128, 1),
                                              torch.nn.Dropout(p=0.2))

        self.loss = CrossEntropyLoss(reduction="none")

    def forward(self, batch):
        processed_batch = self.process_batch(batch)

        this_batch_max_nodes = max(processed_batch["nodes_length_mb"])
        normalized_batch_adj_mats = torch.FloatTensor(processed_batch["adj_mb"]).to(self.device)[:, :,
                                    :this_batch_max_nodes, :this_batch_max_nodes]

        query = torch.FloatTensor(processed_batch["query_mb"]).to(self.device).view(len(batch), self.max_query_size, -1)
        query_lengths = torch.LongTensor(processed_batch["query_length_mb"]).to(self.device)

        packed_representation = torch.nn.utils.rnn.pack_padded_sequence(query, query_lengths.cpu(),
                                                                        batch_first=True, enforce_sorted=False)

        lstm1_output, _ = self.lstm1(packed_representation)
        _, (query_lasthidden, _) = self.lstm2(lstm1_output)

        final_output = query_lasthidden.transpose(1, 0).reshape(len(batch), -1)
        final_output = self.query_dropout(final_output)

        query_to_node = final_output.unsqueeze(1).repeat(1, this_batch_max_nodes, 1)
        nodes = torch.FloatTensor(processed_batch["nodes_mb"]).to(self.device).view(len(batch), self.max_nodes, -1)[:,
                :this_batch_max_nodes, :]
        node_lengths = torch.LongTensor(processed_batch["nodes_length_mb"]).to(self.device)

        node_mask = torch.arange(this_batch_max_nodes, dtype=torch.long).to(self.device).expand(node_lengths.shape[0],
                                                                                                this_batch_max_nodes) < node_lengths.unsqueeze(
            1)
        node_mask = node_mask.unsqueeze(-1).float()

        nodes *= node_mask
        query_to_node *= node_mask

        nodes = self.node_compress_mlp(nodes)

        nodes = torch.cat([query_to_node, nodes], -1)
        nodes = self.node_mlp(nodes)

        vertex_embeddings = self.gcn(nodes, normalized_batch_adj_mats, mask=node_mask)

        vertex_embeddings = vertex_embeddings.view(len(batch), this_batch_max_nodes, -1)
        final_vertex_embeddings = torch.cat([query_to_node, vertex_embeddings], -1)
        final_vertex_embeddings = self.second_mlp(final_vertex_embeddings)

        final_vertex_embeddings *= node_mask

        bmask = torch.FloatTensor(processed_batch["bmask_mb"]).to(self.device)[:, :, :this_batch_max_nodes]

        final_vertex_embeddings = final_vertex_embeddings.squeeze(-1).unsqueeze(1)

        candidate_embeddings = bmask * final_vertex_embeddings
        cand_unconnected = candidate_embeddings == 0

        cand_n_connections = (1 - cand_unconnected.float()).sum(dim=-1)
        cand_connected = torch.min(cand_n_connections, torch.ones_like(cand_n_connections))

        candidate_embeddings = torch.where(cand_unconnected, torch.ones_like(candidate_embeddings) * -1e8,
                                           candidate_embeddings)

        candidate_embeddings, _ = torch.max(candidate_embeddings, dim=-1)

        answers = torch.LongTensor(processed_batch["answer_positions_mb"]).to(self.device)

        gold_candidate_connected = cand_connected[torch.arange(cand_connected.size(0)), answers]

        # This is a bit hacky, might want to refactor.
        # We only see negative targets at test time when the answer is not a mention, so we could actually skip
        # computing the loss entirely in those cases.
        loss_targets = torch.max(answers, torch.zeros_like(answers))
        loss = (self.loss(candidate_embeddings, loss_targets) * gold_candidate_connected).mean()

        scores = torch.softmax(candidate_embeddings, dim=-1).detach().cpu().numpy()

        predictions = []
        for i, example in enumerate(batch):
            example_scores = scores[i]
            example_gold = example["answer_position"]

            example_output = ChoiceModelOutput(example_scores, example_gold)
            predictions.append(example_output)

        return loss, predictions

    def get_gnn(self):
        return self.gcn

    def process_batch(self, data_mb):
        answers_mb = [d["answer_position"] for d in data_mb]

        id_mb = [d['id'] for d in data_mb]

        candidates_orig_mb = [d['candidates_orig'] for d in data_mb]
        candidates_orig_mb2 = [d['candidates_orig2'] for d in data_mb]

        candidates_mb = [d['candidates'] for d in data_mb]

        nodes_mb = np.array([np.pad(np.array([c.mean(0) for c in d['nodes_glove']]),
                                    ((0, self.max_nodes - len(d['nodes_candidates_id'])), (0, 0)),
                                    mode='constant')
                             for d in data_mb])

        query_mb = np.stack([np.pad(d['query_glove'],
                                    ((0, self.max_query_size - d['query_glove'].shape[0]), (0, 0)),
                                    mode='constant')
                             for d in data_mb], 0)

        nodes_length_mb = np.stack([len(d['nodes_candidates_id']) for d in data_mb], 0)
        query_length_mb = np.stack([d['query_glove'].shape[0] for d in data_mb], 0)

        adj_mb = []
        for d in data_mb:

            adj_ = []

            if len(d['edges_in']) == 0:
                adj_.append(np.zeros((self.max_nodes, self.max_nodes)))
            else:
                adj = scipy.sparse.coo_matrix((np.ones(len(d['edges_in'])), np.array(d['edges_in']).T),
                                              shape=(self.max_nodes, self.max_nodes)).toarray()

                adj_.append(adj)

            if len(d['edges_out']) == 0:
                adj_.append(np.zeros((self.max_nodes, self.max_nodes)))
            else:
                adj = scipy.sparse.coo_matrix((np.ones(len(d['edges_out'])), np.array(d['edges_out']).T),
                                              shape=(self.max_nodes, self.max_nodes)).toarray()

                adj_.append(adj)

            if len(d['edges_coref']) == 0:
                adj_.append(np.zeros((self.max_nodes, self.max_nodes)))
            else:
                adj = scipy.sparse.coo_matrix((np.ones(len(d['edges_coref'])), np.array(d['edges_coref']).T),
                                              shape=(self.max_nodes, self.max_nodes)).toarray()

                adj_.append(adj)

            adj = np.pad(np.ones((len(d['nodes_candidates_id']), len(d['nodes_candidates_id']))),
                         ((0, self.max_nodes - len(d['nodes_candidates_id'])),
                          (0, self.max_nodes - len(d['nodes_candidates_id']))), mode='constant') \
                  - adj_[0] - adj_[1] - adj_[2] - np.pad(np.eye(len(d['nodes_candidates_id'])),
                                                         ((0, self.max_nodes - len(d['nodes_candidates_id'])),
                                                          (0, self.max_nodes - len(d['nodes_candidates_id']))),
                                                         mode='constant')

            adj_.append(np.clip(adj, 0, 1))

            adj = np.stack(adj_, 0)

            d_ = adj.sum(-1)
            d_[np.nonzero(d_)] **= -1
            adj = adj * np.expand_dims(d_, -1)

            adj_mb.append(adj)

        adj_mb = np.array(adj_mb)

        bmask_mb = np.array([np.pad(np.array([i == np.array(d['nodes_candidates_id'])
                                              for i in range(len(d['candidates']))]),
                                    ((0, self.max_candidates - len(d['candidates'])),
                                     (0, self.max_nodes - len(d['nodes_candidates_id']))), mode='constant')
                             for d in data_mb])

        return {'id_mb': id_mb, 'nodes_mb': nodes_mb, 'nodes_length_mb': nodes_length_mb,
                'query_mb': query_mb, 'query_length_mb': query_length_mb, 'bmask_mb': bmask_mb,
                'adj_mb': adj_mb, 'candidates_mb': candidates_mb, 'candidates_orig_mb': candidates_orig_mb,
                'candidates_orig_mb2': candidates_orig_mb2, "answer_positions_mb": answers_mb}
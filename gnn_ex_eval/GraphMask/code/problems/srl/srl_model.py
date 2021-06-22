from torch.nn import Embedding, CrossEntropyLoss
import torch

from code.abstract.abstract_torch_module import AbstractTorchModule
from code.gnns.srl_gcn import SrlGcn
from code.problems.srl import srl_utils
from code.problems.srl.srl_utils import parse_word_embeddings, constrained_decoder
from code.problems.srl.voc import create_voc
from code.utils.evaluation.srl_model_output import SrlModelOutput
from code.utils.pad import pad_to_max
import numpy as np

from code.utils.torch_utils.gnn_utils import add_backwards_edges


class SrlModel(AbstractTorchModule):

    def __init__(self, configuration):
        AbstractTorchModule.__init__(self)

        self.configuration = configuration

        pos_index = create_voc('file', configuration["task"]["voc_folder"] + "/pos.voc.conll2009")
        word_index = create_voc('file', configuration["task"]["voc_folder"] + "/words.voc_unk.conll2009")
        word_index.add_unks()
        frame_index = create_voc('file', configuration["task"]["voc_folder"] + "/frames.voc.conll2009")
        role_index = create_voc('file', configuration["task"]["voc_folder"] + "/labels.voc.conll2009")

        self.pretrained_word_embedding_tensor = parse_word_embeddings(
            configuration["task"]["voc_folder"] + "/word_embeddings_proper.sskip.conll2009.txt")
        print("Warning: Rolling pretrained word embeddings to fix indexes")
        self.pretrained_word_embedding_tensor = np.roll(self.pretrained_word_embedding_tensor, 1, axis=0)

        if configuration["task"]["use_lstm"]:
            self.lstm = torch.nn.LSTM(input_size=317,
                                      hidden_size=512,
                                      num_layers=4,
                                      batch_first=True,
                                      bidirectional=True,
                                      dropout=0)
        else:
            self.input_transform = torch.nn.Sequential(
                torch.nn.Linear(317, 2 * 512),
                torch.nn.LayerNorm(2 * 512),
                torch.nn.ReLU()
            )

        self.gnn = SrlGcn(dim=512 * 2,
                          n_layers=configuration["model_parameters"]["gnn_layers"],
                          n_relations=(len(srl_utils._DEP_LABELS) - 1))

        self.role_mlp = torch.nn.Sequential(
            torch.nn.Linear(128 + 128, 2 * 2 * 512),
            torch.nn.ReLU()
        )

        self.loss = CrossEntropyLoss(reduction="none")

        self.word_embedding = Embedding(word_index.size(), 100)
        self.pos_embedding = Embedding(pos_index.size(), 16)
        self.predicate_lemma_embedding = Embedding(frame_index.size(), 100)

        self.pretrained_word_embedding = Embedding(self.pretrained_word_embedding_tensor.shape[0],
                                                   self.pretrained_word_embedding_tensor.shape[1])
        self.pretrained_word_embedding.weight = torch.nn.Parameter(
            torch.from_numpy(self.pretrained_word_embedding_tensor))

        self.frame_embedding = Embedding(frame_index.size(), 128)
        self.role_embedding = Embedding(role_index.size(), 128)

    def forward(self, batch):
        if self.training:
            word_dropout_probabilities = [x["word_dropout_probabilities"] for x in batch]
            padded_word_dropout_probabilities, _ = pad_to_max(word_dropout_probabilities, padding_variable=0.0)
            padded_word_dropout_probabilities = torch.FloatTensor(padded_word_dropout_probabilities).to(self.device)
            bernoulli = torch.empty_like(padded_word_dropout_probabilities).uniform_(0, 1)

            word_dropout_mask = bernoulli > padded_word_dropout_probabilities

        word_indexes = [x["word_indexes"] for x in batch]
        padded_word_indexes, word_lengths = pad_to_max(word_indexes, padding_variable=0)
        padded_word_indexes = torch.LongTensor(padded_word_indexes).to(self.device)

        if self.training:
            padded_word_indexes = torch.where(word_dropout_mask, padded_word_indexes,
                                              torch.zeros_like(padded_word_indexes))

        word_embeddings = self.word_embedding(padded_word_indexes)

        pos_indexes = [x["pos_indexes"] for x in batch]
        padded_pos_indexes, pos_lengths = pad_to_max(pos_indexes, padding_variable=0)
        padded_pos_indexes = torch.LongTensor(padded_pos_indexes).to(self.device)
        pos_embeddings = self.pos_embedding(padded_pos_indexes)

        predicate_lemma_indexes = [x["predicate_lemma_indexes"] for x in batch]
        padded_predicate_lemma_indexes, predicate_lemma_lengths = pad_to_max(predicate_lemma_indexes,
                                                                             padding_variable=0)
        padded_predicate_lemma_indexes = torch.LongTensor(padded_predicate_lemma_indexes).to(self.device)
        predicate_lemma_embeddings = self.predicate_lemma_embedding(padded_predicate_lemma_indexes)

        pretrained_word_indexes = [x["pretrained_word_embedding_index"] for x in batch]
        padded_p_word_indexes, p_word_lengths = pad_to_max(pretrained_word_indexes, padding_variable=0)
        padded_p_word_indexes = torch.LongTensor(padded_p_word_indexes).to(self.device)

        if self.training:
            padded_p_word_indexes = torch.where(word_dropout_mask, padded_p_word_indexes,
                                                torch.zeros_like(padded_p_word_indexes))

        pretrained_word_embeddings = self.pretrained_word_embedding(
            padded_p_word_indexes).detach()  # Detach so we guarantee it is frozen

        region_mark = [x["region_mark"] for x in batch]
        padded_region_marks, region_mark_lengths = pad_to_max(region_mark, padding_variable=0.0)
        padded_region_marks = torch.FloatTensor(padded_region_marks.astype(np.float32)).to(self.device).unsqueeze(-1)

        sentence_embedding = torch.cat([word_embeddings,
                                        pretrained_word_embeddings,
                                        pos_embeddings,
                                        predicate_lemma_embeddings,
                                        padded_region_marks], dim=2)

        sentence_lengths = torch.LongTensor(word_lengths).to(self.device)

        if self.configuration["task"]["use_lstm"]:
            packed_representation = torch.nn.utils.rnn.pack_padded_sequence(sentence_embedding, sentence_lengths.cpu(),
                                                                            batch_first=True, enforce_sorted=False)
            output, _ = self.lstm(packed_representation)
            states, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        else:
            states = self.input_transform(sentence_embedding)

        joint_edge_types = torch.cat(
            [torch.LongTensor(example["dependency_labels"]).to(self.device) for example in batch], 0)

        joint_edges = []
        for i, example in enumerate(batch):
            edge_list = torch.LongTensor(example["dependency_arcs"]).to(self.device) + (states.shape[1] * i)
            joint_edges.append(edge_list)

        joint_edges = torch.cat(joint_edges, 0)

        if joint_edges.shape[0] == 0:
            joint_edges = torch.empty((0, 2), dtype=torch.long, device=self.device)

        joint_edges = joint_edges.transpose(1, 0)

        states_shape = states.shape

        assert joint_edge_types.shape[0] == joint_edges.shape[1]
        if joint_edge_types.shape[0] > 0:
            assert joint_edge_types.max() < 62
            assert joint_edge_types.min() >= 0
            assert joint_edges.max() < states_shape[0] * states_shape[1]
            assert joint_edges.min() >= 0

        use_edges, _, use_edge_types, direction_cutoff = add_backwards_edges(joint_edges,
                                                                             joint_edge_types)
        states = self.gnn(vertex_embeddings=states.view(-1, states.shape[-1]),
                          edges=use_edges,
                          edge_types=use_edge_types,
                          edge_direction_cutoff=direction_cutoff)

        states = states.view(states_shape)

        target_predicate_indexes = [x["target_predicate_idx"] for x in batch]
        target_predicate_indexes = torch.LongTensor(target_predicate_indexes).to(self.device)

        target_predicate_embeddings = states[torch.arange(0, states.shape[0]).to(self.device), target_predicate_indexes,
                                      :]
        target_predicate_embeddings = target_predicate_embeddings.unsqueeze(1).repeat(1, states.shape[1], 1)

        combined_states = torch.cat([states, target_predicate_embeddings], dim=2)

        frame_indexes = [x["frame_indexes"] for x in batch]
        padded_frame_indexes, _ = pad_to_max(frame_indexes, padding_variable=0)
        padded_frame_indexes = torch.LongTensor(padded_frame_indexes).to(self.device)
        frame_embeddings = self.frame_embedding(padded_frame_indexes)

        role_indexes = [x["role_indexes"] for x in batch]
        padded_role_indexes, role_lengths = pad_to_max(role_indexes, padding_variable=0)
        padded_role_indexes = torch.LongTensor(padded_role_indexes).to(self.device)
        role_embeddings = self.role_embedding(padded_role_indexes)

        role_representations = torch.cat([frame_embeddings, role_embeddings], -1)
        role_representations = self.role_mlp(role_representations)

        role_representations = role_representations.transpose_(1, 2)

        scores = torch.bmm(combined_states, role_representations)

        role_lengths = torch.LongTensor(role_lengths).to(self.device)
        max_role_length = role_embeddings.shape[1]
        role_mask = torch.arange(max_role_length).to(self.device).expand(role_lengths.shape[0],
                                                                         max_role_length) >= role_lengths.unsqueeze(1)
        role_mask = role_mask.float() * -1e8
        score_mask = role_mask.unsqueeze(1)

        masked_scores = scores + score_mask

        labels = [x["labels"] for x in batch]
        padded_labels, _ = pad_to_max(labels, padding_variable=0)
        padded_labels = torch.LongTensor(padded_labels).to(self.device)

        flat_masked_scores = masked_scores.view(-1, masked_scores.shape[-1])
        flat_padded_labels = padded_labels.view(-1)
        loss = self.loss(flat_masked_scores, flat_padded_labels)
        loss = loss.view(padded_labels.shape[0], padded_labels.shape[1])

        max_sentence_length = sentence_embedding.shape[1]
        sentence_mask = torch.arange(max_sentence_length).to(self.device).expand(sentence_lengths.shape[0],
                                                                                 max_sentence_length) < sentence_lengths.unsqueeze(
            1)

        # Normalize loss -- first over time, then over batch dimension.
        loss_scaling = sentence_lengths.float()
        loss = torch.where(sentence_mask, loss, torch.zeros_like(loss)).sum(dim=-1) / torch.max(loss_scaling,
                                                                                                torch.ones_like(
                                                                                                    loss_scaling))
        loss = loss.mean()

        predictions = torch.softmax(masked_scores, dim=2)

        outputs = []
        for i, length in enumerate(sentence_lengths):
            local_role_vocabulary = batch[i]["local_role_vocabulary"]
            example_predictions = predictions[i, :length, :len(local_role_vocabulary)].detach().cpu().numpy()

            str_labels = [local_role_vocabulary[l] for l in labels[i]]
            best_labeling = constrained_decoder(local_role_vocabulary, example_predictions, 100, [])

            sentence = []
            n = 2
            for word, prediction, true_label, best, dep_parse in zip(batch[i]["words"],
                                                                     example_predictions,
                                                                     batch[i]["labels"],
                                                                     best_labeling,
                                                                     batch[i]["dep_parse"]):
                nbest = sorted(range(len(prediction)),
                               key=lambda x: -prediction[x])

                nbest = nbest[:n]
                probs = [prediction[l] for l in nbest]

                n_best_labels = [local_role_vocabulary[label] for label in nbest if label in local_role_vocabulary]
                n_best_labels = ' '.join(n_best_labels)

                sentence.append((
                                word, best, n_best_labels, probs, local_role_vocabulary[true_label], batch[i]["degree"],
                                dep_parse))

            example_output = SrlModelOutput(best_labeling, str_labels, sentence)
            outputs.append(example_output)

        return loss, outputs

    def get_gnn(self):
        return self.gnn

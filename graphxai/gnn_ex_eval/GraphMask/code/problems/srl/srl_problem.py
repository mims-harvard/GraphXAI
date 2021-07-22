from functools import partial

from code.problems.srl import srl_utils
from code.problems.srl.corpus import *
from code.problems.srl.srl_utils import *
from code.problems.srl.voc import *
from code.abstract.abstract_test_problem import AbstractTestProblem
import numpy as np

from code.utils.evaluation.srl_evaluator import SrlEvaluator
from code.utils.glove_embedder import GloveEmbedder


class SrlProblem(AbstractTestProblem):

    glove_embedder = None
    cache_location = None
    split_indexes = None

    def __init__(self, configuration):
        self.configuration = configuration

        train_set = Corpus(
            parser=self.get_parser(),
            batch_size=configuration["training"]["batch_size"],
            path=configuration["task"]["train_dataset"],
            reader=self.get_reader()
        )

        dev_set = Corpus(
            parser=self.get_parser(),
            batch_size=configuration["training"]["batch_size"],
            path=configuration["task"]["dev_dataset"],
            reader=self.get_reader()
        )

        test_set = Corpus(
            parser=self.get_parser(),
            batch_size=configuration["training"]["batch_size"],
            path=configuration["task"]["test_dataset"],
            reader=self.get_reader()
        )

        self.raw_train_data = [b for b in train_set.examples()]
        self.raw_dev_data = [b for b in dev_set.examples()]
        self.raw_test_data = [b for b in test_set.examples()]

        self.evaluator = SrlEvaluator(self.configuration)

        self.word_voc = create_voc('file', configuration["task"]["voc_folder"] + "/words.voc_unk.conll2009")
        self.word_voc.add_unks()
        self.freq_voc = frequency_voc(configuration["task"]["voc_folder"] + "/freq.voc_unk.conll2009")
        self.p_word_voc = create_voc('file', configuration["task"]["voc_folder"] + "/p.words.voc_sskip.conll2009")
        self.p_word_voc.add_unks()
        self.role_voc = create_voc('file', configuration["task"]["voc_folder"] + "/labels.voc.conll2009")
        self.frame_voc = create_voc('file', configuration["task"]["voc_folder"] + "/frames.voc.conll2009")
        self.pos_voc = create_voc('file', configuration["task"]["voc_folder"] + "/pos.voc.conll2009")

        self.glove_embedder = GloveEmbedder(self.configuration["preprocessing"]["glove_embeddings"]["file"])

        self.rm_param = 0

        AbstractTestProblem.__init__(self, configuration)

    def count_raw_examples(self, split):
        if split == "train":
            return len(self.raw_train_data)
        elif split == "test":
            return len(self.raw_test_data)
        else:
            return len(self.raw_dev_data)

    def get_parser(self):
        return partial(bio_reader)

    def get_reader(self):
        return simple_reader

    def build_example(self, id, split):
        use_index = id

        if split == "train":
            example = self.raw_train_data[use_index]
        elif split == "test":
            example = self.raw_test_data[use_index]
        else:
            example = self.raw_dev_data[use_index]

        ex = {}
        ex["raw"] = example

        ex["degree"] = example[1][4]
        ex["dep_parse"] = example[1][3]
        ex["words"] = example[1][1]

        ex["diego_input"] = self.bio_converter([example[1]])

        ex["word_indexes"] = ex["diego_input"][0][0]
        ex["pos_indexes"] = ex["diego_input"][2][0]
        ex["predicate_lemma_indexes"] = ex["diego_input"][10][0]
        ex["pretrained_word_embedding_index"] = ex["diego_input"][1][0]

        ex["region_mark"] = ex["diego_input"][9][0]

        # Handle division before caching for very minor speedup
        # Word dropout prob: alpha / (#(word) + alpha)
        word_dropout_alpha = 0.25
        ex["word_dropout_probabilities"] = word_dropout_alpha / (ex["diego_input"][8][0] + word_dropout_alpha)

        ex["target_predicate_idx"] = ex["diego_input"][4][0]

        ex["frame_indexes"] = ex["diego_input"][5][0]
        ex["role_indexes"] = ex["diego_input"][6][0]

        ex["labels"] = ex["diego_input"][18][0]
        ex["role_mask"] = ex["diego_input"][7][0]

        dep_arcs = []
        dep_labels = []
        for label, source, target in example[1][3]:
            if label != "ROOT" and label in srl_utils._DEP_LABELS:
                # Ignore bad edges -- the dataset has one or two
                if source - 1 < 0 or target - 1 < 0 or source - 1 >= len(ex["word_indexes"]) or target - 1 >= len(
                        ex["word_indexes"]):
                    continue

                assert source - 1 >= 0
                assert source - 1 < len(ex["word_indexes"])
                assert target - 1 >= 0
                assert target - 1 < len(ex["word_indexes"])
                dep_arcs.append([source - 1, target - 1])
                dep_labels.append(srl_utils._DEP_LABELS[label] - 1)  # -1 because we do not use ROOT, which has index 0.

        if len(dep_arcs) == 0:
            dep_arcs = np.empty((0, 2), dtype=np.int32)

        dep_arcs = np.array(dep_arcs, dtype=np.int32)
        dep_labels = np.array(dep_labels, dtype=np.int32)

        ex["dependency_arcs"] = dep_arcs
        ex["dependency_labels"] = dep_labels

        ex["local_role_vocabulary"] = {i: self.role_voc.get_item(x) for i, x in enumerate(ex["role_indexes"])}

        example = ex

        return example

    def bio_converter(self, batch):
        headers, sent_, pos_tags, dep_parsing, degree, frames, \
        targets, f_lemmas, f_targets, labels_voc, labels = list(
            zip(*batch))

        sent = [self.word_voc.vocalize(w) for w in sent_]
        p_sent = [self.p_word_voc.vocalize(w) for w in sent_]
        freq = [[self.freq_voc[self.word_voc.direct[i]] if
                 self.word_voc.direct[i] != '_UNK' else 0 for i in w] for
                w
                in sent]

        pos_tags = [self.pos_voc.vocalize(w) for w in pos_tags]
        frames = [self.frame_voc.vocalize(f) for f in frames]
        labels_voc = [self.role_voc.vocalize(r) for r in labels_voc]

        lemmas_idx = [self.frame_voc.vocalize(f) for f in f_lemmas]

        adj_arcs_in, adj_arcs_out, adj_lab_in, adj_lab_out, \
        mask_in, mask_out, mask_loop = get_adj(dep_parsing, degree)

        sent_batch, sent_mask = mask_batch(sent)
        p_sent_batch, _ = mask_batch(p_sent)
        freq_batch, _ = mask_batch(freq)
        freq_batch = freq_batch.astype(dtype='float32')

        pos_batch, _ = mask_batch(pos_tags)
        labels_voc_batch, labels_voc_mask = mask_batch(labels_voc)
        labels_batch, _ = mask_batch(labels)
        frames_batch, _ = mask_batch(frames)

        region_mark = np.zeros(sent_batch.shape, dtype='float32')

        rm = self.rm_param
        if rm >= 0:
            for r, row in enumerate(region_mark):
                for c, column in enumerate(row):
                    if targets[r] - rm <= c <= targets[r] + rm:
                        region_mark[r][c] = 1

        sent_pred_lemmas_idx = np.zeros(sent_batch.shape, dtype='int32')
        for r, row in enumerate(sent_pred_lemmas_idx):
            for c, column in enumerate(row):
                for t, tar in enumerate(f_targets[r]):
                    if tar == c:
                        sent_pred_lemmas_idx[r][c] = lemmas_idx[r][t]

        sent_pred_lemmas_idx = np.array(sent_pred_lemmas_idx, dtype='int32')

        assert (sent_batch.shape == sent_mask.shape)
        assert (
                frames_batch.shape == labels_voc_batch.shape == labels_voc_mask.shape)
        assert (labels_batch.shape == sent_batch.shape)
        return sent_batch, p_sent_batch, pos_batch, sent_mask, targets, frames_batch, \
               labels_voc_batch, \
               labels_voc_mask, freq_batch, \
               region_mark, \
               sent_pred_lemmas_idx, \
               adj_arcs_in, adj_arcs_out, adj_lab_in, adj_lab_out, \
               mask_in, mask_out, mask_loop, \
               labels_batch

    def overwrite_labels(self, batch, predictions):
        updated_batch = []

        for example, prediction in zip(batch, predictions):
            role_vocab = example["local_role_vocabulary"]
            label_dic = {v:k for k,v in role_vocab.items()}

            new_labels = np.array([label_dic[x] for x in prediction.get_predictions()], dtype=np.int32)

            updated_example = example.copy()
            updated_example["labels"] = new_labels

            updated_batch.append(updated_example)

        return updated_batch
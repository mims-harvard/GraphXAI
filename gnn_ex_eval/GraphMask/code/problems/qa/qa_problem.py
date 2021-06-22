import json
import os
import pickle
import re
import sys

import numpy as np
from allennlp.predictors.predictor import Predictor
from spacy.lang.en import English
from code.abstract.abstract_test_problem import AbstractTestProblem
from code.utils.evaluation.choice_evaluator import ChoiceEvaluator
from code.utils.glove_embedder import GloveEmbedder


class QAProblem(AbstractTestProblem):
    n_edge_types = 4
    max_nodes = None
    max_query_size = None
    max_candidates = None

    def __init__(self, configuration):
        self.max_nodes = configuration["task"]["max_nodes"]
        self.max_query_size = configuration["task"]["max_query_size"]
        self.max_candidates = configuration["task"]["max_candidates"]
        dataset_location = configuration["task"]["dataset_folder"]

        in_file = dataset_location + "wikihop/train.json"
        with open(in_file, 'r') as f:
            self.raw_train_data = json.load(f)

        in_file = dataset_location + "wikihop/dev.json"
        with open(in_file, 'r') as f:
            self.raw_dev_data = json.load(f)

        in_file = dataset_location + "wikihop/dev.json"
        with open(in_file, 'r') as f:
            self.raw_test_data = json.load(f)

        self.nlp = English()
        self.glove_embedder = GloveEmbedder(configuration["preprocessing"]["glove_embeddings"]["file"])
        # self.predictor = Predictor.from_path(
        #    "https://allennlp.s3.amazonaws.com/models/coref-model-2020.02.10.tar.gz")

        # self.predictor = pretrained.load_predictor("coref")
        self.predictor = Predictor.from_path(
            "https://s3-us-west-2.amazonaws.com/allennlp/models/coref-model-2018.02.05.tar.gz")
        # self.predictor = pretrained.load_predictor("coref-spanbert")
        # self.predictor = Predictor.from_path(
        #    "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-base-2020.02.27.tar.gz")

        self.evaluator = ChoiceEvaluator()

        AbstractTestProblem.__init__(self, configuration)

    def skip_example(self, d, split):
        if "answer_position" not in d:
            #print("Warning: Skipping an example because something went wrong when constructing the answer.", file=sys.stderr)
            return True

        if split == "train":
            if len(d['nodes_candidates_id']) > self.max_nodes:
                #print("Warning: Skipping an example because I generated too many nodes.", file=sys.stderr)
                return True

            if len(d['query']) > self.max_query_size:
                #print("Warning: Skipping an example because the query is too long.", file=sys.stderr)
                return True

            if len(d['candidates']) > self.max_candidates:
                #print("Warning: Skipping an example because there are too many candidates.", file=sys.stderr)
                return True

            if len(d['nodes_candidates_id']) == 0:
                #print("Warning: Skipping an example because there are no candidates.", file=sys.stderr)
                return True

            if d["answer_position"] == -1:
                #print("Warning: Skipping an example because the answer was not a mention.", file=sys.stderr)
                return True

        return False

    def count_raw_examples(self, split):
        if split == "train":
            return len(self.raw_train_data)
        elif split == "test":
            return len(self.raw_test_data)
        else:
            return len(self.raw_dev_data)

    def build_example(self, id, split):
        use_index = id

        if split == "train":
            example = self.raw_train_data[use_index]
        elif split == "test":
            example = self.raw_test_data[use_index]
        else:
            example = self.raw_dev_data[use_index]

        processed_example = self.process_example(example)

        return processed_example

    def regex(self, text):
        text = text.replace(u'\xa0', ' ')
        text = text.translate(str.maketrans({key: ' {0} '.format(key) for key in '"!&()*+,/:;<=>?[]^`{|}~'}))
        text = re.sub('\s{2,}', ' ', text).replace('\n', '')

        return text

    def check(self, s, wi, c):
        return sum([s[wi + j].lower() == c_ for j, c_ in enumerate(c) if wi + j < len(s)]) == len(c)

    def ind(self, si, wi, ci, c):
        return [[si, wi + i, ci] for i in range(len(c))]

    def compute_coref(self, s):
        try:
            ret = self.predictor.predict(s)
            return ret['clusters'], ret['document']
        except RuntimeError:
            return [], [str(w) for w in self.nlp(s)]

    def process_example(self, d):
        d['candidates_orig'] = list(d['candidates'])
        d['candidates'] = [c for c in d['candidates'] if
                           c not in self.nlp.Defaults.stop_words]  # This + cand in support?
        d['candidates_orig2'] = list(d['candidates'])
        d['candidates'] = [[str(w) for w in c] for c in self.nlp.pipe(d['candidates'])]
        d['query'] = [str(w) for w in self.nlp.tokenizer(d['query'])]
        d['supports'] = [self.regex(s) for s in d['supports']]
        tmp = [self.compute_coref(s) for s in d['supports']]

        d['supports'] = [e for _, e in tmp]
        d['coref'] = [e for e, _ in tmp]
        d['coref'] = [[[[f, []] for f in e] for e in s]
                      for s in d['coref']]
        mask = [[self.ind(si, wi, ci, c) for wi, w in enumerate(s)
                 for ci, c in enumerate(d['candidates'] + [d['query'][1:]])
                 if self.check(s, wi, c)] for si, s in enumerate(d['supports'])]

        nodes = []
        for sc, sm in zip(d['coref'], mask):
            u = []
            for ni, n in enumerate(sm):
                k = []
                for cli, cl in enumerate(sc):

                    x = [(n[0][1] <= co[0] <= n[-1][1]) or (co[0] <= n[0][1] <= co[1])
                         for co, cll in cl]

                    for i, v in filter(lambda y: y[1], enumerate(x)):
                        k.append((cli, i))
                        cl[i][1].append(ni)
                u.append(k)
            nodes.append(u)

        # remove one entity with multiple coref
        for sli, sl in enumerate(nodes):
            for ni, n in enumerate(sl):
                if len(n) > 1:
                    for e0, e1 in n:
                        i = d['coref'][sli][e0][e1][1].index(ni)
                        del d['coref'][sli][e0][e1][1][i]
                    sl[ni] = []

        # remove one coref with multiple entity
        for ms, cs in zip(nodes, d['coref']):
            for cli, cl in enumerate(cs):
                for eli, (el, li) in enumerate(cl):
                    if len(li) > 1:
                        for e in li:
                            i = ms[e].index((cli, eli))
                            del ms[e][i]
                        cl[eli][1] = []
        d['edges_coref'] = []
        for si, (ms, cs) in enumerate(zip(mask, d['coref'])):
            tmp = []
            for cl in cs:
                cand = {ms[n[0]][0][-1] for p, n in cl if n}
                if len(cand) == 1:
                    cl_ = []
                    for (p0, p1), _ in cl:
                        if not _:
                            cl_.append(len(ms))
                            ms.append([[si, i, list(cand)[0]] for i in range(p0, p1 + 1)])
                        else:
                            cl_.append(_[0])
                    tmp.append(cl_)
            d['edges_coref'].append(tmp)
        nodes_id_name = []
        c = 0
        for e in [[[x[-1] for x in c][0] for c in s] for s in mask]:
            u = []
            for f in e:
                u.append((c, f))
                c += 1

            nodes_id_name.append(u)
        d['nodes_candidates_id'] = [[x[-1] for x in f][0] for e in mask for f in e]
        edges_in, edges_out = [], []
        for e0 in nodes_id_name:
            for f0, w0 in e0:
                for f1, w1 in e0:
                    if f0 != f1:
                        edges_in.append((f0, f1))

                for e1 in nodes_id_name:
                    for f1, w1 in e1:
                        if e0 != e1 and w0 == w1:
                            edges_out.append((f0, f1))
        edges_coref = []
        for nins, cs in zip(nodes_id_name, d['edges_coref']):
            for cl in cs:
                for e0 in cl:
                    for e1 in cl:
                        if e0 != e1:
                            edges_coref.append((nins[e0][0], nins[e1][0]))
        d['edges_coref'] = edges_coref
        d['edges_in'] = edges_in
        d['edges_out'] = edges_out
        d['edges'] = edges_in + edges_out + edges_coref
        mask_ = [[x[:-1] for x in f] for e in mask for f in e]

        candidate_glove = self.glove_embedder.batch_to_embeddings(d["supports"])
        d['nodes_glove'] = [(candidate_glove[tuple(np.array(m).T.tolist())]).astype(np.float16)
                            for m in mask_]
        d['nodes_glove'] = d['nodes_glove'][:self.max_nodes]

        d['nodes_candidates_id'] = d['nodes_candidates_id'][:self.max_nodes]
        d['edges_in'] = [e for e in d['edges_in'] if e[0] < self.max_nodes and e[1] < self.max_nodes]
        d['edges_out'] = [e for e in d['edges_out'] if e[0] < self.max_nodes and e[1] < self.max_nodes]
        d['edges_coref'] = [e for e in d['edges_coref'] if e[0] < self.max_nodes and e[1] < self.max_nodes]

        d["query_glove"] = self.glove_embedder.batch_to_embeddings([d['query']])[0]
        d['query_glove'] = d['query_glove'][:self.max_query_size]

        orig_cands = d['candidates_orig']
        orig_cands = [c for c in orig_cands if c not in self.nlp.Defaults.stop_words]
        orig_cands = [c for c in orig_cands if any(c.lower() in " ".join(supp).lower() for supp in d["supports"])]

        if d["answer"] in orig_cands:
            d["answer_position"] = orig_cands.index(d["answer"])
        else:
            d["answer_position"] = -1

        return d

    def overwrite_labels(self, batch, predictions):
        new_batch = []
        for i in range(len(batch)):
            example = batch[i]
            prediction = predictions[i].get_prediction()

            copied_example = example.copy()
            copied_example["answer_position"] = int(prediction)

            new_batch.append(copied_example)

        return new_batch

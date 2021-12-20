"""
util functions copied from the original codebase at https://github.com/diegma/neural-dep-srl/
"""
import heapq
import math
from collections import namedtuple

import numpy as np
import sys

def mask_batch(batch):
    max_len = len(max(batch, key=len))
    mask = np.zeros((len(batch), max_len))
    padded = np.zeros((len(batch), max_len))
    for i in range(len(batch)):
        mask[i, :len(batch[i])] = 1
        for j in range(len(batch[i])):
            padded[i, j] = batch[i][j]

    return padded.astype('int32'), mask.astype('int32')

def make_local_voc(labels):
    return {i: label for i, label in enumerate(labels)}

def bio_reader(record):
    dbg_header, sent, pos_tags, dep_parsing, degree, frame, target, f_lemmas, f_targets, labels_voc, labels = record.split(
        '\t')
    labels_voc = labels_voc.split(' ')

    frame = [frame] * len(labels_voc)
    words = []
    for word in sent.split(' '):
        words.append(word)

    pos_tags = pos_tags.split(' ')
    labels = labels.split(' ')

    assert (len(words) == len(labels))

    local_voc = {v: k for k, v in make_local_voc(labels_voc).items()}
    labels = [local_voc[label] for label in labels]

    dep_parsing = dep_parsing.split()
    dep_parsing = [p.split('|') for p in dep_parsing]
    dep_parsing = [(p[0], int(p[1]), int(p[2])) for p in dep_parsing]

    f_lemmas = f_lemmas.split(' ')
    f_targets = f_targets.split(' ')
    return dbg_header, words, pos_tags, dep_parsing, np.int32(degree), frame, \
           np.int32(target), f_lemmas, np.int32(f_targets), labels_voc, labels



# conll2009 labels
_DEP_LABEL_LIST = ['ROOT', 'ADV', 'ADV-GAP', 'AMOD', 'APPO', 'BNF', 'CONJ', 'COORD', 'DEP',
               'DEP-GAP', 'DIR', 'DIR-GAP', 'DIR-OPRD', 'DIR-PRD', 'DTV', 'EXT',
               'EXT-GAP', 'EXTR', 'GAP-LGS', 'GAP-LOC', 'GAP-LOC-PRD', 'GAP-MNR',
               'GAP-NMOD', 'GAP-OBJ', 'GAP-OPRD', 'GAP-PMOD', 'GAP-PRD', 'GAP-PRP',
               'GAP-SBJ', 'GAP-TMP', 'GAP-VC', 'HMOD', 'HYPH', 'IM', 'LGS', 'LOC',
               'LOC-OPRD', 'LOC-PRD', 'LOC-TMP', 'MNR', 'MNR-PRD', 'MNR-TMP', 'NAME',
               'NMOD', 'OBJ', 'OPRD', 'P', 'PMOD', 'POSTHON', 'PRD', 'PRD-PRP',
               'PRD-TMP', 'PRN', 'PRP', 'PRT', 'PUT', 'SBJ', 'SUB', 'SUFFIX',
               'TITLE', 'TMP', 'VC', 'VOC']

_DEP_LABELS = {label: i for i, label in enumerate(_DEP_LABEL_LIST)}

_N_LABELS = len(_DEP_LABELS)

def get_adj(deps, degree):
    batch_size = len(deps)
    _MAX_BATCH_LEN = 0

    for de in deps:
        if len(de) > _MAX_BATCH_LEN:
            _MAX_BATCH_LEN = len(de)

    _MAX_DEGREE = max(degree)

    adj_arc_in = np.zeros((batch_size * _MAX_BATCH_LEN, 2), dtype='int32')
    adj_lab_in = np.zeros((batch_size * _MAX_BATCH_LEN, 1), dtype='int32')
    adj_arc_out = np.zeros((batch_size * _MAX_BATCH_LEN*_MAX_DEGREE, 2), dtype='int32')
    adj_lab_out = np.zeros((batch_size * _MAX_BATCH_LEN*_MAX_DEGREE, 1), dtype='int32')


    mask_in = np.zeros((batch_size * _MAX_BATCH_LEN), dtype='float32')
    mask_out = np.zeros((batch_size * _MAX_BATCH_LEN * _MAX_DEGREE), dtype='float32')

    mask_loop = np.ones((batch_size * _MAX_BATCH_LEN, 1), dtype='float32')

    tmp_in = {}
    tmp_out = {}

    for d, de in enumerate(deps):
        for a, arc in enumerate(de):
            if arc[0] != 'ROOT' and arc[0] in _DEP_LABELS:
                arc_1 = int(arc[1])-1
                arc_2 = int(arc[2])-1
                if a in tmp_in:
                    tmp_in[a] += 1
                else:
                    tmp_in[a] = 0

                if arc_2 in tmp_out:
                    tmp_out[arc_2] += 1
                else:
                    tmp_out[arc_2] = 0

                idx_in = (d * _MAX_BATCH_LEN) + a + tmp_in[a]
                idx_out = (d * _MAX_BATCH_LEN * _MAX_DEGREE) + arc_2 * _MAX_DEGREE + tmp_out[arc_2]

                adj_arc_in[idx_in] = np.array([d, arc_2])  # incoming arcs
                adj_lab_in[idx_in] = np.array([_DEP_LABELS[arc[0]]])  # incoming arcs

                mask_in[idx_in] = 1.

                if tmp_out[arc_2] < _MAX_DEGREE:
                    adj_arc_out[idx_out] = np.array([d, arc_1])  # outgoing arcs
                    adj_lab_out[idx_out] = np.array([_DEP_LABELS[arc[0]]])  # outgoing arcs
                    mask_out[idx_out] = 1.

        tmp_in = {}
        tmp_out = {}

    return np.transpose(adj_arc_in), np.transpose(adj_arc_out), \
           np.transpose(adj_lab_in), np.transpose(adj_lab_out), \
           mask_in.reshape((len(deps) * _MAX_BATCH_LEN, 1)), \
           mask_out.reshape((len(deps) * _MAX_BATCH_LEN, _MAX_DEGREE)), \
           mask_loop


def parse_word_embeddings(embeddings):
    res = []

    for line in open(embeddings, 'r'):
        emb = map(float, line.strip().split()[1:])
        res.append(list(emb))

    return np.array(res, dtype='float32')

def _get_score(state):
    return state.score

def _continuation_constraint_no_bio(state):
    if not state.label.startswith('C-'):
        return True

    if state.label[2:] in state.roles:
        return True

    return False

CONLL2009_CONSTRAINTS = []
_PRUNING_THRESHOLD = 5.
State = namedtuple('State', ['score', 'label', 'prev', 'roles'])

def constrained_decoder(voc, predictions, beam, constraints):
    heap = [State(score=0, label='O', prev=None, roles=set())]
    for i, prediction in enumerate(predictions):
        next_generation = list()
        for prev in heapq.nsmallest(beam, heap, key=_get_score):
            for j, prob in enumerate(prediction):
                label = voc[j]
                score = -math.log2(prob + sys.float_info.min)
                if score > _PRUNING_THRESHOLD and next_generation:
                    continue

                next_state = State(score=score + prev.score,
                                   label=label, prev=prev,
                                   roles=prev.roles)

                constraints_violated = [not check(next_state) for check in
                                        constraints]
                if any(constraints_violated):
                    continue

                next_generation.append(
                    State(next_state.score, next_state.label, next_state.prev,
                          next_state.roles | {next_state.label[2:]}))

        heap = next_generation

    head = heapq.nsmallest(1, heap, key=_get_score)[0]

    backtrack = list()
    while head:
        backtrack.append(head.label)
        head = head.prev

    return list(reversed(backtrack[:-1]))
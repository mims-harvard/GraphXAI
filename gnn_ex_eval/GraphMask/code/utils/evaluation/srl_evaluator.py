import sys

from subprocess import check_output
import os
import sys

import numpy as np

class SrlEvaluator:

    """
    Evaluator for Diego Marcheggiani's BioSRL.

    Fast scoring uses macro-f1 on the example level, slow scoring uses the official eval script.
    """

    mode = None
    eval_dir = None

    def __init__(self, configuration):
        self.eval_dir = configuration["task"]["eval_dir"]
        self.eval_file = configuration["task"]["id"] + ".out"

    def is_perfect_score(self, predictions):
        score = self.score_batch(predictions)

        return score == self.get_optimal_performance()

    def compare_performance(self, old_performance, new_performance):
        return old_performance < new_performance

    def score_batch(self, predictions):
        scores = [self.score_example(p) for p in predictions]
        return float(sum(scores)) / max(len(scores), 1)

    def score_example(self, prediction):
        actual_preds = prediction.get_predictions()
        labels = prediction.get_labels()

        tp_by_class = {}
        fp_by_class = {}
        fn_by_class = {}
        for i in range(len(labels)):
            if actual_preds[i] == labels[i]:
                if actual_preds[i] not in tp_by_class:
                    tp_by_class[actual_preds[i]] = 1
                else:
                    tp_by_class[actual_preds[i]] += 1
            elif actual_preds[i] != labels[i]:
                if actual_preds[i] not in fp_by_class:
                    fp_by_class[actual_preds[i]] = 1
                else:
                    fp_by_class[actual_preds[i]] += 1

                if labels[i] not in fn_by_class:
                    fn_by_class[labels[i]] = 1
                else:
                    fn_by_class[labels[i]] += 1

        keys = set(list(tp_by_class.keys()) + list(fp_by_class.keys()) + list(fn_by_class.keys()))

        mean_f1 = 0
        for c in keys:
            tp = tp_by_class[c] if c in tp_by_class else 0.0
            fp = fp_by_class[c] if c in fp_by_class else 0.0
            fn = fn_by_class[c] if c in fn_by_class else 0.0

            precision = tp / (tp + fp) if tp + fp > 0 else 0.0
            recall = tp / (tp + fn) if tp + fn > 0 else 1.0
            f1 = 2 * precision * recall / (precision + recall) if tp > 0 else 0.0

            mean_f1 += f1

        mean_f1 /= len(keys)

        return mean_f1

    def get_optimal_performance(self):
        return 1.0

    def evaluate_stats(self, all_stats, split):
        data_partition = split

        results = self.do_eval(data_partition,
                               all_stats,
                               self.eval_dir)

        return results[2]

    def get_stats(self, prediction):
        return prediction.get_eval_line()

    def get_evaluation_metric_name(self):
        return "span_f1"

    def set_mode(self, mode):
        self.mode = mode

    def iterate_sentences(self, dataset_filename):
        current_sentence = []
        with open(dataset_filename, 'r') as f:
            for line in f:
                line = line.strip()

                if not line:
                    yield current_sentence
                    current_sentence = []
                else:
                    current_sentence.append(line.split("\t"))

        if len(current_sentence) > 0:
            yield current_sentence

    def do_eval(self, subset_name, prediction, data_dir):

        def get_path(name):
            return os.path.join(data_dir, name)

        def get_proper_format(separate_predicate_list):
            with open(self.eval_file + '.prediction_09.txt', 'w') as out:
                sent = ''
                sent_annotation = []
                for pred in separate_predicate_list:
                    if sent == ' '.join([a[0] for a in pred]):
                        annotation = []
                        for t in range(len(pred)):
                            label = pred[t][1]
                            if label == 'O':
                                label = '_'
                            if t == pred[0][-2]:
                                annotation.append(('Y', label))
                            else:
                                annotation.append(('_', label))
                        sent_annotation[-1].append(
                            (annotation, pred[0][-2], pred[0][-1]))
                    else:
                        if sent != '':
                            sent_annotation[-1] = sorted(sent_annotation[-1],
                                                         key=lambda x: int(x[1]))

                            for i, token in enumerate(sent_annotation[-1][0][0]):
                                single_token = []
                                for j in range(len(sent_annotation[-1])):
                                    single_token.append(
                                        sent_annotation[-1][j][0][i][1])

                                out.write('\t'.join(single_token) + '\n')
                            out.write('\n')

                        annotation = []
                        sent = ' '.join([a[0] for a in pred])
                        for t in range(len(pred)):
                            label = pred[t][1]
                            if label == 'O':
                                label = '_'
                            if t == pred[0][-2]:
                                annotation.append(('Y', label))
                            else:
                                annotation.append(('_', label))
                        sent_annotation.append(
                            [(annotation, pred[0][-2], pred[0][-1])])

                sent_annotation[-1] = sorted(sent_annotation[-1],
                                             key=lambda x: int(x[1]))
                for i, token in enumerate(sent_annotation[-1][0][0]):
                    single_token = []
                    for j in range(len(sent_annotation[-1])):
                        single_token.append(sent_annotation[-1][j][0][i][1])

                    out.write('\t'.join(single_token) + '\n')
                out.write('\n')

        get_proper_format(prediction)

        script = 'aux/official_scripts/conll09/eval09.pl'
        gold_data = get_path('%s-set_for_eval_gold' % subset_name)

        eval_script_args = " ".join([script, '-g', gold_data, '-s ', self.eval_file + '.prediction_09.paste'])

        if True:
            pred_sentence_generator = self.iterate_sentences(get_path('%s-set_for_eval_ppred' % subset_name))
            pred_generator = self.iterate_sentences(self.eval_file + '.prediction_09.txt')

            with open(self.eval_file + '.prediction_09.paste', 'w') as f:
                has_printed = False
                while True:
                    if has_printed:
                        print("\n", file=f, end="")

                    try:
                        next_pred_sentence = next(pred_sentence_generator)
                    except StopIteration:
                        break

                    has_preds = any([x[12] == "Y" for x in next_pred_sentence])
                    if not has_preds:
                        output = "\n".join(["\t".join(token) for token in next_pred_sentence])
                        print(output, file=f)
                        has_printed = True
                        continue

                    try:
                        next_pred = next(pred_generator)
                    except StopIteration:
                        print("Error: Test file longer than pred file.")
                        exit()

                    combined = [x + y for x, y in zip(next_pred_sentence, next_pred)]

                    output = "\n".join(["\t".join(token) for token in combined])
                    print(output, file=f)
                    has_printed = True

            DEVNULL = open(os.devnull, 'wb')

            #past_script = ['paste', get_path('%s-set_for_eval_ppred' % subset_name),
            #               'prediction_09.txt']

            #out_paste = check_output(past_script, stderr=DEVNULL)

            #out_paste = out_paste.decode('utf-8')
            #open('prediction_09.paste', 'w').write(out_paste)

            print(eval_script_args)

            out = check_output(eval_script_args, stderr=DEVNULL, shell=True)
            out = out.decode('utf-8')

            open(self.eval_file + '.eval09.out', 'w').write(out)
            results = out.strip().split('\n')
            precision = results[7].split(' %')[0].split('= ')[1]
            recall = results[8].split(' %')[0].split('= ')[1]
            f1 = results[9].split()[2]
            print('--------------------------------------------------',
                  file=sys.stderr)
            print('Official script results:', file=sys.stderr)
            print('Precision: ' + precision, 'recall: ' + recall, 'F1: ' + f1,
                  file=sys.stderr)
            print('--------------------------------------------------',
                  file=sys.stderr)
            return float(precision), float(recall), float(f1)
        else:
            print('--------------------------------------------------',
                  file=sys.stderr)
            print('There has been some error with the official script',
                  file=sys.stderr)
            print('Try next iteration :) ', file=sys.stderr)
            print('--------------------------------------------------',
                  file=sys.stderr)
            return float(0.0), float(0.0), float(0.0)

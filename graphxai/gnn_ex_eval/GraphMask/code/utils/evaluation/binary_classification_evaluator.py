import numpy as np

class BinaryClassificationEvaluator:

    """
    Evaluator for classifying n elements with binary labels.

    Fast scoring uses accuracy, slow scoring uses f1-score of the positive class.
    """

    def is_perfect_score(self, predictions):
        score = self.score_batch(predictions)

        return score == self.get_optimal_performance()

    def score_batch(self, predictions):
        scores = [self.score_example(p) for p in predictions]
        return float(sum(scores)) / max(len(scores), 1)

    def score_example(self, prediction):
        actual_preds = prediction.get_predictions()
        labels = prediction.get_labels()

        accuracy = float(sum(p == i for p, i in zip(actual_preds, labels))) / len(labels)

        return accuracy

    def get_optimal_performance(self):
        return 1.0

    def evaluate_stats(self, all_stats, split):
        all_stats = np.array(all_stats)
        all_stats = all_stats.sum(axis=0)
        tp, fp, fn = all_stats

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 1.0
        f1 = 2 * precision * recall / (precision + recall) if tp > 0 else 0.0

        return f1

    def set_mode(self, mode):
        self.mode = mode

    def get_stats(self, prediction):
        tp = 0
        fp = 0
        fn = 0

        actual_preds = prediction.get_predictions()
        labels = prediction.get_labels()

        for i in range(len(labels)):
            if actual_preds[i] == labels[i] and labels[i] == 1:
                tp += 1
            elif actual_preds[i] != labels[i] and labels[i] == 0:
                fp += 1
            elif actual_preds[i] != labels[i] and labels[i] == 1:
                fn += 1

        return tp, fp, fn

    def get_evaluation_metric_name(self):
        return "macro_f1"

    def compare_performance(self, old_performance, new_performance):
        return old_performance < new_performance
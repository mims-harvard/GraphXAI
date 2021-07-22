import numpy as np

class ChoiceEvaluator:

    """
    Evaluator for classifying one elements with n labels.

    Fast and slow scoring both use accuracy.
    """

    def score_batch(self, predictions):
        scores = [self.score_example(p) for p in predictions]
        return float(sum(scores)) / max(len(scores), 1)

    def compare_performance(self, old_performance, new_performance):
        return old_performance < new_performance

    def score_example(self, prediction):
        return float(prediction.get_prediction() == prediction.get_gold_answer())

    def get_optimal_performance(self):
        return 1.0

    def evaluate_stats(self, all_stats, split):
        all_stats = np.array(all_stats)

        return all_stats.mean()

    def get_stats(self, prediction):
        return [self.score_example(prediction)]

    def get_evaluation_metric_name(self):
        return "accuracy"

    def set_mode(self, mode):
        pass
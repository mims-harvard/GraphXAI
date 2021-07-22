import torch
from tqdm import tqdm
import numpy as np

class AnalysisRunner:

    def __init__(self, configuration):
        self.configuration = configuration

    def disable_all_gradients(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def fit_analyser(self, model, problem, analysis_technique, do_validate=True, gpu_number=-1):
        device = torch.device('cuda:' + str(gpu_number) if torch.cuda.is_available() and gpu_number >= 0 else 'cpu')
        model.set_device(device)

        analysis_technique.initialise_for_model(model, problem)

        self.disable_all_gradients(model)

        analysis_technique.fit(model, problem, gpu_number=gpu_number)

        if do_validate:
            analysis_technique.validate(model, problem, gpu_number=gpu_number)

    def match_gold_standard(self, model, problem, analysis_technique, gpu_number=-1):
        problem.evaluator.set_mode("test")
        device = torch.device('cuda:' + str(gpu_number) if torch.cuda.is_available() and gpu_number >= 0 else 'cpu')
        model.set_device(device)

        batch_size = 1

        model.eval()
        problem.initialize_epoch()
        batch_iterator = tqdm(problem.iterate_batches(batch_size=batch_size, split="test"),
                              total=problem.approximate_batch_count(batch_size=batch_size, split="test"),
                              dynamic_ncols=True,
                              smoothing=0.0)

        all_stats = []

        for i, batch in enumerate(batch_iterator):
            message_attributions = analysis_technique.analyse(batch, model, problem)

            for p, e in zip(message_attributions, batch):
                stats = problem.score_attributions(p, e)
                all_stats.append(stats)

        all_stats = np.array(all_stats)
        all_stats = all_stats.sum(axis=0)
        tp, fp, fn = all_stats

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 1.0
        f1 = 2 * precision * recall / (precision + recall) if tp > 0 else 0.0

        return precision, recall, f1

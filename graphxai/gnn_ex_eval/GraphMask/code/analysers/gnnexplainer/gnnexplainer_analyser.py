import sys

import numpy as np
import torch
from tqdm import tqdm

from code.analysers.gnnexplainer.gnnexplainer_probe import GNNExplainerProbe
from code.analysers.graphmask.graphmask_probe import GraphMaskProbe
from code.analysers.graphmask.graphmask_unamortised_probe import GraphMaskUnamortisedProbe
from code.utils.moving_average import MovingAverage
from code.utils.torch_utils.lagrangian_optimization import LagrangianOptimization


class GNNExplainerAnalyser:

    probe = None
    moving_average_window_size = 100

    def __init__(self, configuration):
        self.configuration = configuration

    def initialise_for_model(self, model, problem):
        pass

    def validate(self, model, problem, split="test", gpu_number=-1):
        #threshold = self.configuration["analysis"]["parameters"]["threshold"]
        threshold = 0.1
        problem.evaluator.set_mode(split)

        batch_size = 1

        model.eval()
        problem.initialize_epoch()
        score_moving_average = MovingAverage(window_size=self.moving_average_window_size)
        sparsity_moving_average = MovingAverage(window_size=self.moving_average_window_size)

        batch_iterator = tqdm(problem.iterate_batches(batch_size=batch_size, split=split),
                              total=problem.approximate_batch_count(batch_size=batch_size, split=split),
                              dynamic_ncols=True,
                              smoothing=0.0)

        original_all_stats = []
        gated_all_stats = []

        all_gates = 0
        all_messages = 0

        for i, batch in enumerate(batch_iterator):
            _, original_predictions = model(batch)
            for p, e in zip(original_predictions, batch):
                original_score = problem.evaluator.score_example(p)
                stats = problem.evaluator.get_stats(p)

                original_all_stats.append(stats)

            probe = self.initialise_for_batch(batch, model, problem)
            self.fit_probe(probe, batch, gpu_number, model, problem)
            probe.eval()

            gates, _ = probe()

            model.get_gnn().inject_message_scale(gates)
            _, predictions = model(batch)

            gates = gates > threshold

            for p, e in zip(predictions, batch):
                gated_score = problem.evaluator.score_example(p)
                stats = problem.evaluator.get_stats(p)

                gated_all_stats.append(stats)

            score_diff = abs(float(gated_score - original_score))
            score_moving_average.register(score_diff)

            all_gates += float(sum([g.sum().detach() for g in gates]))
            all_messages += float(model.get_gnn().count_latest_messages())
            batch_sparsity = float(sum([g.sum().detach() for g in gates]) / model.get_gnn().count_latest_messages())
            sparsity_moving_average.register(batch_sparsity)

            batch_iterator.set_description("Evaluation mean score difference={0:.4f}, mean retained={1:.4f}".format(
                score_moving_average.get_value(),
                sparsity_moving_average.get_value()))

        original_true_score = problem.evaluator.evaluate_stats(original_all_stats, split)
        gated_true_score = problem.evaluator.evaluate_stats(gated_all_stats, split)

        print("GNNExplainer comparison on the " + split + "-split:")
        print("======================================")
        print("Original test score: " + str(original_true_score))
        print("Gated test score: " + str(gated_true_score))
        print("Retained messages: " + str(all_gates / all_messages))

        diff = np.abs(original_true_score - gated_true_score)
        percent_div = float(diff / (original_true_score + 1e-8))

        sparsity = float(all_gates / all_messages)

        return percent_div, sparsity

    def fit(self, model, problem, gpu_number=-1):
        pass

    def compute_loss(self, probe, batch, model, problem):
        model.eval()
        _, original_predictions = model(batch)

        model.train()  # Enable any dropouts in the original model. We found this helpful for training GraphMask.
        probe.train()

        batch = problem.overwrite_labels(batch, original_predictions)

        gates, penalty = probe()

        model.get_gnn().inject_message_scale(gates)
        loss, predictions = model(batch)

        return loss, predictions, penalty

    def initialise_for_batch(self, batch, model, problem):
        num_layers = model.get_gnn().count_layers()
        num_edges = problem.count_gnn_input_edges(batch)

        return GNNExplainerProbe(num_edges, num_layers)

    def analyse(self, batch, model, problem, gpu_number=-1):
        #threshold = self.configuration["analysis"]["parameters"]["threshold"]
        threshold = 0.1

        probe = self.initialise_for_batch(batch, model, problem)

        self.fit_probe(probe, batch, gpu_number, model, problem)

        probe.eval()
        gates, _ = probe()

        gates = gates > threshold

        return gates

    def fit_probe(self, probe, batch, gpu_number, model, problem):
        penalty_scaling = self.configuration["analysis"]["parameters"]["penalty_scaling"] # 0.001 works well
        learning_rate = self.configuration["analysis"]["parameters"]["learning_rate"]
        epochs = self.configuration["analysis"]["parameters"]["epochs_per_layer"]

        penalty_scaling = 0.001
        epochs = 2000
        optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
        device = torch.device('cuda:' + str(gpu_number) if torch.cuda.is_available() and gpu_number >= 0 else 'cpu')
        probe.set_device(device)

        for epoch in range(epochs):
            probe.train()
            optimizer.zero_grad()
            loss, predictions, penalty = self.compute_loss(probe, batch, model, problem)

            total_loss = loss + penalty * penalty_scaling
            total_loss.backward()

            optimizer.step()

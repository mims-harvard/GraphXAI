import sys

import numpy as np
import torch
from tqdm import tqdm

from code.analysers.graphmask.graphmask_probe import GraphMaskProbe
from code.analysers.information_bottleneck.information_bottleneck_probe import InformationBottleneckProbe
from code.utils.moving_average import MovingAverage
from code.utils.torch_utils.lagrangian_optimization import LagrangianOptimization


class InformationBottleneckAnalyser:

    probe = None
    moving_average_window_size = 100

    def __init__(self, configuration):
        self.configuration = configuration

    def obtain_message_mean_and_var(self, model, problem):
        batch_size = 1

        stat_tensor = []

        with torch.no_grad():
            model.eval()
            problem.initialize_epoch()
            batch_iterator = tqdm(problem.iterate_batches(batch_size=batch_size, split="dev"),
                                  total=problem.approximate_batch_count(batch_size=batch_size, split="dev"),
                                  dynamic_ncols=True,
                                  smoothing=0.0,
                                  desc="Retrieving message statistics from the dev set.")

            for i, batch in enumerate(batch_iterator):
                model(batch)

                latest_messages = np.array([m.detach().numpy() for m in model.get_gnn().get_latest_messages()])
                stat_tensor.append(latest_messages)

        stat_tensor = np.concatenate(stat_tensor, axis=1)

        return np.array([stat_tensor.mean(axis=1), stat_tensor.var(axis=1)])

    def initialise_for_model(self, model, problem):
        vertex_embedding_dims = model.get_gnn().get_vertex_embedding_dims()
        message_dims = model.get_gnn().get_message_dims()

        message_mean_and_var = self.obtain_message_mean_and_var(model, problem)

        self.probe = InformationBottleneckProbe(vertex_embedding_dims, message_dims, message_dims, message_mean_and_var)

    def validate(self, model, problem, split="test", gpu_number=-1):
        threshold = self.configuration["analysis"]["parameters"]["threshold"]
        problem.evaluator.set_mode(split)

        device = torch.device('cuda:' + str(gpu_number) if torch.cuda.is_available() and gpu_number >= 0 else 'cpu')
        self.probe.set_device(device)
        model.set_device(device)

        batch_size = 1

        with torch.no_grad():
            model.eval()
            self.probe.eval()
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

                gates, baselines, _ = self.probe(model.get_gnn())
                model.get_gnn().inject_message_scale(gates)
                model.get_gnn().inject_message_replacement(baselines)
                _, predictions = model(batch)

                gates = [g > threshold for g in gates]
                for p, e in zip(predictions, batch):
                    gated_score = problem.evaluator.score_example(p)
                    stats = problem.evaluator.get_stats(p)

                    gated_all_stats.append(stats)

                score_diff = abs(float(gated_score - original_score))
                score_moving_average.register(score_diff)

                all_gates += float(sum([g.sum().detach() for g in gates]))
                all_messages += float(model.get_gnn().count_latest_messages())
                batch_sparsity = float(sum([g.sum().detach() for g in gates])/model.get_gnn().count_latest_messages())
                sparsity_moving_average.register(batch_sparsity)

                batch_iterator.set_description("Evaluation mean score difference={0:.4f}, mean retained={1:.4f}".format(
                    score_moving_average.get_value(),
                    sparsity_moving_average.get_value()))

            original_true_score = problem.evaluator.evaluate_stats(original_all_stats, split)
            gated_true_score = problem.evaluator.evaluate_stats(gated_all_stats, split)

            print("Information bottleneck comparison on the "+split+"-split:")
            print("======================================")
            print("Original test score: " + str(original_true_score))
            print("Gated test score: " + str(gated_true_score))
            print("Retained messages: " + str(all_gates / all_messages))

            diff = np.abs(original_true_score - gated_true_score)
            percent_div = float(diff / (original_true_score + 1e-8))

            sparsity = float(all_gates / all_messages)

            return percent_div, sparsity

    def fit(self, model, problem, gpu_number=-1):
        batch_size = self.configuration["analysis"]["parameters"]["batch_size"]
        epochs_per_layer = self.configuration["analysis"]["parameters"]["epochs_per_layer"]
        train_split = self.configuration["analysis"]["parameters"]["train_split"]
        test_every_n = self.configuration["analysis"]["parameters"]["test_every_n"]
        save_path = self.configuration["analysis"]["parameters"]["save_path"]
        penalty_scaling = self.configuration["analysis"]["parameters"]["penalty_scaling"]
        learning_rate = self.configuration["analysis"]["parameters"]["learning_rate"]
        allowance = self.configuration["analysis"]["parameters"]["allowance"]
        max_allowed_performance_diff = self.configuration["analysis"]["parameters"]["max_allowed_performance_diff"]
        load = self.configuration["analysis"]["parameters"]["load"]
        train = self.configuration["analysis"]["parameters"]["train"]

        if load:
            self.probe.load(save_path)

        if train:
            if "batch_size_multiplier" in self.configuration["analysis"]["parameters"] and \
                    self.configuration["analysis"]["parameters"]["batch_size_multiplier"] > 1:
                batch_size_multiplier = self.configuration["analysis"]["parameters"]["batch_size_multiplier"]
            else:
                batch_size_multiplier = None

            optimizer = torch.optim.Adam(self.probe.parameters(), lr=learning_rate)

            device = torch.device('cuda:' + str(gpu_number) if torch.cuda.is_available() and gpu_number >= 0 else 'cpu')
            self.probe.set_device(device)
            model.set_device(device)

            # Lagrange optimization is not originally a part of the IB approach, but we can get much more consistent results by using it
            lagrangian_optimization = LagrangianOptimization(optimizer,
                                                             device,
                                                             batch_size_multiplier=batch_size_multiplier)

            f_moving_average = MovingAverage(window_size=self.moving_average_window_size)
            g_moving_average = MovingAverage(window_size=self.moving_average_window_size)

            best_sparsity = 1.01
            for epoch in range(epochs_per_layer):
                problem.evaluator.set_mode("train")
                problem.initialize_epoch()
                batch_iterator = tqdm(problem.iterate_batches(batch_size=batch_size, split=train_split),
                                      total=problem.approximate_batch_count(batch_size=batch_size,
                                                                            split=train_split),
                                      dynamic_ncols=True,
                                      smoothing=0.0)

                for i, batch in enumerate(batch_iterator):
                    self.probe.train()
                    loss, predictions, penalty = self.compute_loss(batch, model, problem)

                    g = torch.relu(loss - allowance).mean()
                    f = penalty * penalty_scaling

                    lagrangian_optimization.update(f, g)

                    f_moving_average.register(float(f))
                    g_moving_average.register(float(loss.mean()))

                    batch_iterator.set_description(
                        "Running epoch {0:n} of GraphMask training. Mean divergence={1:.4f}, mean penalty={2:.4f}".format(
                            epoch,
                            g_moving_average.get_value(),
                            f_moving_average.get_value()))

                if (epoch + 1) % test_every_n == 0:
                    percent_div, sparsity = self.validate(model, problem, split="dev", gpu_number=gpu_number)

                    if percent_div < max_allowed_performance_diff and sparsity < best_sparsity:
                        print("Found better probe with sparsity={0:.4f}. Keeping these parameters.".format(sparsity),
                              file=sys.stderr)
                        best_sparsity = sparsity
                        self.probe.save(save_path)

            # Load the best probe:
            self.probe.load(save_path)

    def compute_loss(self, batch, model, problem):
        model.eval()
        _, original_predictions = model(batch)

        model.train() # Enable any dropouts in the original model. We found this helpful for training GraphMask.
        self.probe.train()

        batch = problem.overwrite_labels(batch, original_predictions)

        gates, baselines, penalty = self.probe(model.get_gnn())
        model.get_gnn().inject_message_scale(gates)
        model.get_gnn().inject_message_replacement(baselines)
        loss, predictions = model(batch)

        return loss, predictions, penalty

    def analyse(self, batch, model, problem):
        threshold = self.configuration["analysis"]["parameters"]["threshold"] # 0.3 or 0.4 worked well for us, but it may take a few restarts to get a good solution

        model.eval()
        self.probe.eval()
        _, original_predictions = model(batch)

        gates, _, _ = self.probe(model.get_gnn())

        return [g > threshold for g in gates]

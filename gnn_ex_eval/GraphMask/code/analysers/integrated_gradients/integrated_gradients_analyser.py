import sys

import numpy as np
import torch
from tqdm import tqdm

from code.analysers.graphmask.graphmask_probe import GraphMaskProbe
from code.analysers.graphmask.graphmask_unamortised_probe import GraphMaskUnamortisedProbe
from code.analysers.integrated_gradients.integrated_gradients_probe import IntegratedGradientsProbe
from code.utils.moving_average import MovingAverage
from code.utils.torch_utils.lagrangian_optimization import LagrangianOptimization


class IntegratedGradientsAnalyser:

    probe = None
    moving_average_window_size = 100

    def __init__(self, configuration):
        self.configuration = configuration

    def initialise_for_model(self, model, problem):
        pass

    def validate(self, model, problem, split="test", gpu_number=-1):
        pass

    def fit(self, model, problem, gpu_number=-1):
        pass

    def initialise_for_batch(self, batch, model, problem):
        num_edges = problem.count_gnn_input_edges(batch)
        num_layers = model.gnn.count_layers()

        return IntegratedGradientsProbe(num_edges, num_layers)

    def analyse(self, batch, model, problem, gpu_number=-1):
        threshold = self.configuration["analysis"]["parameters"]["threshold"] #We used 0.55, but best value varies greatly

        device = torch.device('cuda:' + str(gpu_number) if torch.cuda.is_available() and gpu_number >= 0 else 'cpu')
        model.set_device(device)

        probe = self.initialise_for_batch(batch, model, problem)
        probe.set_device(device)

        model.eval()
        _, original_predictions = model(batch)
        batch = problem.overwrite_labels(batch, original_predictions)

        store = torch.zeros_like(probe())
        samples = 40

        probe.train()
        for i in range(0,samples):
            probe.zero_grad()
            import ipdb
            ipdb.set_trace()
            scale = float(i + 1) / float(samples)
            scaled_pseudo_gates = scale * probe()
            model.gnn.inject_message_scale(scaled_pseudo_gates)

            loss, predictions = model(batch)
            loss.backward()

            store += (probe().grad / float(samples)).detach()

        bound = max(abs(store.max()), abs(store.min()))
        store /= bound

        important = abs(store) > threshold

        return important
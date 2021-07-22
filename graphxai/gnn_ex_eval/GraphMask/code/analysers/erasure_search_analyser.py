import itertools
import numpy as np
import torch

class ErasureSearchAnalyser:

    def __init__(self, configuration):
        self.configuration = configuration

    def initialise_for_model(self, model):
        pass

    def analyse(self, batch, model, problem, gpu_number=-1):
        _, original_predictions = model(batch)
        batch = problem.overwrite_labels(batch, original_predictions)

        count_messages = model.count_messages(batch)
        count_layers = model.gnn.n_layers
        all_combinations = itertools.product(range(2), repeat=count_messages)

        best_message_scale = None
        best_penalty = None

        device = torch.device('cuda:' + str(gpu_number) if torch.cuda.is_available() and gpu_number >= 0 else 'cpu')
        model.set_device(device)

        for combination in all_combinations:
            message_scale = np.array(combination, dtype=np.float32).reshape((count_layers, -1))
            model.gnn.inject_message_scale(torch.tensor(message_scale).to(device))
            _, new_prediction = model(batch)

            prediction_unchanged = problem.evaluator.is_perfect_score(new_prediction)

            if prediction_unchanged:
                combination_penalty = sum(combination)

                if best_penalty is None or combination_penalty < best_penalty:
                    best_penalty = combination_penalty
                    best_message_scale = message_scale

        return best_message_scale

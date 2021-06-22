import torch
from tqdm import tqdm

from code.utils.moving_average import MovingAverage


class ExperimentRunner:

    moving_average_window_size = 100

    def __init__(self, configuration):
        self.configuration = configuration

    def evaluate_model(self, model, problem, split, gpu_number=-1):
        problem.evaluator.set_mode(split)
        device = torch.device('cuda:' + str(gpu_number) if torch.cuda.is_available() and gpu_number >= 0 else 'cpu')
        model.set_device(device)

        batch_size = 1

        with torch.no_grad():
            model.eval()
            problem.initialize_epoch()
            score_moving_average = MovingAverage(window_size=self.moving_average_window_size)
            batch_iterator = tqdm(problem.iterate_batches(batch_size=batch_size, split=split),
                                  total=problem.approximate_batch_count(batch_size=batch_size, split=split),
                                  dynamic_ncols=True,
                                  smoothing=0.0)

            all_stats = []

            for i, batch in enumerate(batch_iterator):
                _, predictions = model(batch)
                for p, e in zip(predictions, batch):
                    score = problem.evaluator.score_example(p)
                    score_moving_average.register(float(score))
                    stats = problem.evaluator.get_stats(p)

                    all_stats.append(stats)

                batch_iterator.set_description("Evaluation mean score={0:.4f})".format(
                    score_moving_average.get_value()))

            true_score = problem.evaluator.evaluate_stats(all_stats, split)
            print("True dev score: " + str(true_score))

        return true_score

    def train_model(self, model, problem, gpu_number=-1):
        batch_size = self.configuration["training"]["batch_size"]
        max_epochs = self.configuration["training"]["max_epochs"]
        train_split = self.configuration["training"]["train_split"]
        test_every_n = self.configuration["training"]["test_every_n"]
        save_path = self.configuration["training"]["save_path"]
        learning_rate = self.configuration["training"]["learning_rate"]

        if "batch_size_multiplier" in self.configuration["training"] and self.configuration["training"]["batch_size_multiplier"] > 1:
            batch_size_multiplier = self.configuration["training"]["batch_size_multiplier"]
            update_counter = 0
        else:
            batch_size_multiplier = None

        device = torch.device('cuda:' + str(gpu_number) if torch.cuda.is_available() and gpu_number >= 0 else 'cpu')
        model.set_device(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        best_dev_performance = None

        for epoch in range(max_epochs):
            problem.evaluator.set_mode("train")
            problem.initialize_epoch()
            loss_moving_average = MovingAverage(window_size=self.moving_average_window_size)
            score_moving_average = MovingAverage(window_size=self.moving_average_window_size)
            batch_iterator = tqdm(problem.iterate_batches(batch_size=batch_size, split=train_split),
                                  total=problem.approximate_batch_count(batch_size=batch_size, split=train_split),
                                  dynamic_ncols=True,
                                  smoothing=0.0)

            for i, batch in enumerate(batch_iterator):
                model.train()

                if batch_size_multiplier is not None:
                    if update_counter % batch_size_multiplier == 0:
                        optimizer.zero_grad()

                    update_counter += 1
                else:
                    optimizer.zero_grad()

                loss, predictions = model(batch)
                loss = loss.mean()
                loss.backward()

                if batch_size_multiplier is not None:
                    if update_counter % batch_size_multiplier == 0:
                        clipping_value = 1
                        torch.nn.utils.clip_grad_value_(model.parameters(), clipping_value)

                        optimizer.step()
                else:
                    clipping_value = 1
                    torch.nn.utils.clip_grad_value_(model.parameters(), clipping_value)

                    optimizer.step()

                loss_moving_average.register(float(loss.detach()))

                pred_score = problem.evaluator.score_batch(predictions)
                score_moving_average.register(float(pred_score))

                batch_iterator.set_description("Epoch " + str(epoch) + " (mean loss={0:.4f}, mean score={1:.4f})".format(
                                                   loss_moving_average.get_value(),
                                                   score_moving_average.get_value()))

            if (epoch + 1) % test_every_n == 0:
                dev_performance = self.evaluate_model(model, problem, "dev", gpu_number=gpu_number)

                if best_dev_performance is None or problem.evaluator.compare_performance(best_dev_performance, dev_performance):
                    best_dev_performance = dev_performance
                    model.save(save_path)

        model.load(save_path)
        return model
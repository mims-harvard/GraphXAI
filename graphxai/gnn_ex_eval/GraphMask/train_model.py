import argparse

from code.utils.configuration_loader import ConfigurationLoader
from code.utils.runners.experiment_runner import ExperimentRunner
from code.utils.experiment_utils import ExperimentUtils


class ModelTrainer:

    configuration = None
    gpu = None
    log_file = None
    model_location = None

    def __init__(self, configuration, gpu):
        self.configuration = configuration
        self.gpu = gpu
        self.model_location = configuration["task"]["id"]

        self.experiment_utils = ExperimentUtils(configuration)

    def train(self):
        problem = self.experiment_utils.build_problem()
        model = self.experiment_utils.build_model()

        experiment_runner = ExperimentRunner(self.configuration)

        experiment_runner.train_model(model, problem, gpu_number=self.gpu)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train according to a specified configuration file.')
    parser.add_argument("--gpu", default=-1, type=int)
    parser.add_argument("--configuration", default="configurations/star_graphs.json")
    args = parser.parse_args()

    configuration_loader = ConfigurationLoader()
    configuration = configuration_loader.load(args.configuration)

    model_trainer = ModelTrainer(configuration, gpu=args.gpu)
    model_trainer.train()
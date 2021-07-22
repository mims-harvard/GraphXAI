import argparse

import ipdb
from code.utils.configuration_loader import ConfigurationLoader
from code.utils.experiment_utils import ExperimentUtils
from code.utils.runners.analysis_runner import AnalysisRunner


class ModelAnalyser:

    def __init__(self, configuration, gpu):
        self.configuration = configuration
        self.gpu = gpu
        self.model_location = configuration["task"]["id"]

        self.experiment_utils = ExperimentUtils(configuration)

    def match_gold_standard(self):
        problem = self.experiment_utils.build_problem()
        model = self.experiment_utils.load_trained_model()
        analysis_technique = self.experiment_utils.build_analyser()

        analysis_runner = AnalysisRunner(self.configuration)
        analysis_runner.fit_analyser(model, problem, analysis_technique, gpu_number=self.gpu)

        p, r, f1 = analysis_runner.match_gold_standard(model, problem, analysis_technique, gpu_number=self.gpu)

        print("Analysis complete. Gold standard score:")
        print("Precision:\t\t"+str(p))
        print("Recall:\t\t"+str(r))
        print("F1:\t\t\t"+str(f1))

    def compute_analysis(self):
        problem = self.experiment_utils.build_problem()
        model = self.experiment_utils.load_trained_model(gpu_number=self.gpu)
        analysis_technique = self.experiment_utils.build_analyser()

        analysis_runner = AnalysisRunner(self.configuration)
        analysis_runner.fit_analyser(model, problem, analysis_technique, gpu_number=self.gpu)
        # model.eval()
        # problem.initialize_epoch()
        # from tqdm import tqdm
        # batch_size = 1
        # batch_iterator = tqdm(problem.iterate_batches(batch_size=batch_size, split="test"),
        #                       total=problem.approximate_batch_count(batch_size=batch_size, split="test"),
        #                       dynamic_ncols=True,
        #                       smoothing=0.0)
        # for i, batch in enumerate(batch_iterator):
        #     ipdb.set_trace()
        #     message_attributions = analysis_technique.analyse(batch, model, problem)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train according to a specified configuration file.')
    parser.add_argument("--gpu", default=-1, type=int)
    parser.add_argument("--configuration", default="configurations/star_graphs.json")
    args = parser.parse_args()

    configuration_loader = ConfigurationLoader()
    configuration = configuration_loader.load(args.configuration)

    model_analyser = ModelAnalyser(configuration, gpu=args.gpu)
    #model_analyser.match_gold_standard()

    model_analyser.compute_analysis()
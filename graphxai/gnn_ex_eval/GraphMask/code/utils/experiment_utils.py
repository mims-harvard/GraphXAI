import os
import sys
import torch

from code.analysers.erasure_search_analyser import ErasureSearchAnalyser
from code.analysers.gnnexplainer.gnnexplainer_analyser import GNNExplainerAnalyser
from code.analysers.graphmask.graphmask_analyser import GraphMaskAnalyser
from code.analysers.graphmask.graphmask_unamortised_analyser import GraphMaskUnamortisedAnalyser
from code.analysers.information_bottleneck.information_bottleneck_analyser import InformationBottleneckAnalyser
from code.analysers.integrated_gradients.integrated_gradients_analyser import IntegratedGradientsAnalyser
from code.problems.qa.qa_model import QAModel
from code.problems.qa.qa_problem import QAProblem
from code.problems.srl.srl_model import SrlModel
from code.problems.srl.srl_problem import SrlProblem
from code.problems.star_graphs.star_graph_model import StarGraphModel
from code.problems.star_graphs.star_graph_problem import StarGraphProblem


class ExperimentUtils:

    def __init__(self, configuration):
        self.configuration = configuration

    def build_problem(self):
        problem_class_name = self.configuration["task"]["problem_class"]
        if problem_class_name == "StarGraphs":
            problem_class = StarGraphProblem
        elif problem_class_name == "Srl":
            problem_class = SrlProblem
        elif problem_class_name == "QA":
            problem_class = QAProblem

        problem = problem_class(self.configuration)

        return problem

    def build_model(self):
        problem_class_name = self.configuration["task"]["problem_class"]
        if problem_class_name == "StarGraphs":
            model_class = StarGraphModel
        elif problem_class_name == "Srl":
            model_class = SrlModel
        elif problem_class_name == "QA":
            model_class = QAModel

        model = model_class(self.configuration)

        return model

    def build_analyser(self):
        analyser_name = self.configuration["analysis"]["strategy"]
        if analyser_name == "ErasureSearch":
            analyser_class = ErasureSearchAnalyser
        elif analyser_name == "GraphMask":
            analyser_class = GraphMaskAnalyser
        elif analyser_name == "GraphMaskUnamortised":
            analyser_class = GraphMaskUnamortisedAnalyser
        elif analyser_name == "IntegratedGradients":
            analyser_class = IntegratedGradientsAnalyser
        elif analyser_name == "GNNExplainer":
            analyser_class = GNNExplainerAnalyser
        elif analyser_name == "InformationBottleneck":
            analyser_class = InformationBottleneckAnalyser

        analyser = analyser_class(self.configuration)

        return analyser

    def load_trained_model(self, gpu_number):
        model = self.build_model()
        device = torch.device('cuda:' + str(gpu_number) if torch.cuda.is_available() and gpu_number >= 0 else 'cpu')
        model.set_device(device)

        save_path = self.configuration["training"]["save_path"]

        if not os.path.exists(save_path):
            print("Warning: No saved model found, I will assume that this is a debugging run and use random parameters.", file=sys.stderr)
        else:
            model.load(save_path)

        return model


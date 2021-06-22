import numpy as np

class SrlModelOutput:

    labels = None
    predictions = None
    eval_line = None

    def __init__(self, predictions, labels, eval_line):
        self.predictions = predictions
        self.labels = labels
        self.eval_line = eval_line

    def get_predictions(self):
        return self.predictions

    def get_labels(self):
        return self.labels

    def get_eval_line(self):
        return self.eval_line
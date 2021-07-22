import numpy as np

class BinaryClassificationModelOutput:

    labels = None
    scores = None
    predictions = None

    def __init__(self, scores, labels):
        self.predictions = (scores > 0.5).astype(np.bool)
        self.scores = scores
        self.labels = labels.astype(np.bool)

    def get_predictions(self):
        return self.predictions

    def get_labels(self):
        return self.labels
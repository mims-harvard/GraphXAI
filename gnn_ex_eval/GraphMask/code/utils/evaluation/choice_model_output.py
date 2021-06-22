import numpy as np

class ChoiceModelOutput:

    gold_answer = None
    scores = None
    prediction = None

    def __init__(self, scores, gold_answer):
        self.prediction = np.argmax(scores)
        self.scores = scores
        self.gold_answer = gold_answer

    def get_prediction(self):
        return self.prediction

    def get_gold_answer(self):
        return self.gold_answer
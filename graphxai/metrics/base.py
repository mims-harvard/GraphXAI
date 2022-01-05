

class Metrics:

    def __init__(self, model, explainer):
        self.model = model
        self.explainer = explainer
        pass

    # Metric 1-3: Different metrics implemented within the class

    def metric1(self):
        pass

    def metric2(self):
        pass

    def metric3(self):
        pass

    def evaluate(self, name: str = 'all'):
        '''
        Args:
            name (str): Name of metric to evaluate
        '''
        pass
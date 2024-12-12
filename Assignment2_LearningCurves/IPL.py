import numpy as np
from vertical_model_evaluator import VerticalModelEvaluator
from scipy.optimize import curve_fit

class IPL(VerticalModelEvaluator):
    def __init__(self, fixed_schedule, final_anchor, best_seen_performance):
        """
        Initialize the IPLModelEvaluator.

        Args:
            fixed_schedule (list): Fixed training sizes for fitting the IPL.
            anchor_points (list): All possible anchor sizes for evaluation.
            best_seen_performance (float): Current best performance (loss).
            surrogate_model (object): Surrogate model to predict performances.
            minimal_anchor (int): Smallest anchor size.
            final_anchor (int): Largest anchor size.
        """
        self.fixed_schedule = fixed_schedule
        self.max_anchor = final_anchor
        self.best_seen_performance = best_seen_performance
        self.performance_over_iterations = []
        self.last_seen_prediction = None

    def fit_ipl(self, sizes, losses):
        """
        Fit an Inverse Power Law (IPL) model to the observed data.

        Args:
            sizes (list): Training sizes used so far.
            losses (list): Observed losses corresponding to the training sizes.

        Returns:
            params (tuple): Fitted parameters (a, b, c) for the IPL.
        """
        def ipl(x, a, b, c):
            return a * (x ** -b) + c

        params, _ = curve_fit(ipl, sizes, losses, maxfev=100000)
        return params

    def predict_performance(self, size, params):
        """
        Predict the loss for a given size using the fitted IPL.

        Args:
            size (int): Training size to predict the loss for.
            params (tuple): Fitted parameters (a, b, c) for the IPL.

        Returns:
            float: Predicted loss.
        """
        a, b, c = params
        return a * (size ** -b) + c

    def evaluate_configuration(self, performances):
        """
        Evaluate a configuration using the IPL model and fixed schedule.
        Args:
            config (str): Identifier for the configuration.
            losses (list): Observed losses for the fixed schedule.
        Returns:
            bool: True if configuration is promising and should continue, False otherwise.
        """        
        # Fit the IPL model to the observed learning curve
        params = self.fit_ipl(self.fixed_schedule, performances)

        predicted_performance = self.predict_performance(self.max_anchor, params)
        print(f"max_anchor: {self.max_anchor}, predicted_performance: {predicted_performance}, best_seen_performance: {self.best_seen_performance}")
        self.performance_over_iterations.append(self.best_seen_performance)
        self.last_seen_prediction = predicted_performance
        if predicted_performance > self.best_seen_performance:
            return False
        else: 
            return True


        


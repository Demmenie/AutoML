import logging
import numpy as np
import pandas as pd
import scipy.stats as st
import typing

from vertical_model_evaluator import VerticalModelEvaluator

class LCCV(VerticalModelEvaluator):
    def __init__(self, surrogate_model, minimal_anchor: int, final_anchor: int):
        """
        Initializes the LCCV evaluator.

        :param surrogate_model: The surrogate model used for predictions.
        :param minimal_anchor: The smallest anchor size for evaluation.
        :param final_anchor: The largest anchor size for evaluation.
        """
        self.surrogate_model = surrogate_model
        self.minimal_anchor = minimal_anchor
        self.final_anchor = final_anchor
        self.last_seen_opt_ext = None  # Initialize to None for tracking later
        self.best_opt_ext_seen = 1 #float('inf')  # For best optimistic extrapolation tracking

    # @staticmethod
    # def optimistic_extrapolation(
    #     previous_anchor: int, previous_performance: float, 
    #     current_anchor: int, current_performance: float, target_anchor: int
    # ) -> float:
    #     """
    #     Perform optimistic extrapolation of the performance.

    #     :param previous_anchor: Anchor size of the previous stage.
    #     :param previous_performance: Performance at the previous anchor size.
    #     :param current_anchor: Anchor size of the current stage.
    #     :param current_performance: Performance at the current anchor size.
    #     :param target_anchor: Anchor size for extrapolation.
    #     :return: The extrapolated performance.
    #     """
    #     mean = (previous_performance + current_performance) / 2
    #     sd = np.std([previous_performance, current_performance])

    #     if sd == 0.0:
    #         sd = 1e-10  # Prevent division by zero

    #     prevC_i = st.norm.interval(0.95, loc=mean, scale=sd)
    #     C_i = st.norm.interval(0.95, loc=mean, scale=sd)

    #     sub = ((prevC_i[1] - C_i[0]) / (previous_performance - current_performance + 1e-10))
    #     optPerf = C_i[0] - (target_anchor - current_anchor) * sub

    #     return optPerf
    
    @staticmethod
    def optimistic_extrapolation_2(
            previous_anchor: int, previous_performance: float, 
            current_anchor: int, current_performance: float, target_anchor: int
        ) -> float:
        """
        Perform optimistic extrapolation of the performance.

        :param previous_anchor: Anchor size of the previous stage.
        :param previous_performance: Performance at the previous anchor size.
        :param current_anchor: Anchor size of the current stage.
        :param current_performance: Performance at the current anchor size.
        :param target_anchor: Anchor size for extrapolation.
        :return: The extrapolated performance.
        """
        # print("Debugging optimistic_extrapolation_2:")
        # print(f"previous_anchor: {previous_anchor}, previous_performance: {previous_performance}")
        # print(f"current_anchor: {current_anchor}, current_performance: {current_performance}")
        # print(f"target_anchor: {target_anchor}")

        # Handle edge cases to prevent division by zero or unrealistic extrapolation
        if current_anchor == previous_anchor or current_performance == previous_performance:
            # print("No trend observed; returning current performance.")
            return current_performance

        # Calculate the slope of the performance trend
        slope = (current_performance - previous_performance) / (current_anchor - previous_anchor)
        # print(f"slope: {slope}")

        # Extrapolate performance at the target anchor
        extrapolated_performance = current_performance + slope * (target_anchor - current_anchor)
        # print(f"extrapolated_performance before clamping: {extrapolated_performance}")

        # Ensure extrapolated performance is non-negative
        extrapolated_performance = extrapolated_performance
        # print(f"extrapolated_performance after clamping: {extrapolated_performance}")

        return extrapolated_performance
    

    def evaluate_model(self, anchor_sizes: list, best_so_far: typing.Optional[float], configuration: typing.Dict) -> typing.List[typing.Tuple[int, float]]:
        """
        Does a staged evaluation of the model using a predefined array of anchor sizes.
        Determines after each evaluation whether to stop based on extrapolation.
        :param best_so_far: indicates which performance has been obtained so far
        :param configuration: A dictionary indicating the configuration
        :return: A list of tuples, each containing the anchor size and the estimated performance.
        """

        # Predefined array of anchor sizes
        # anchor_sizes = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
        anchor_sequence = []
        x = pd.DataFrame(configuration, index=[0])

        # If best_so_far is None, initialize it with the performance at the largest anchor size
        if best_so_far is None:
            x["anchor_size"] = self.final_anchor
            best_so_far = self.surrogate_model.predict(x, anchor_sizes[-1])

        for current_anchor in anchor_sizes:
            # Add current anchor size to the configuration
            x["anchor_size"] = current_anchor
            pred_perf = self.surrogate_model.predict(x)

            # Log the anchor size and performance
            anchor_sequence.append((current_anchor, pred_perf))

            # If we have at least two anchor points, perform extrapolation
            if len(anchor_sequence) > 1:
                previous_anchor, previous_perf = anchor_sequence[-2]  # Get the last two points
                current_anchor, current_perf = anchor_sequence[-1]

                opt_ext = self.optimistic_extrapolation_2(previous_anchor, previous_perf, current_anchor, current_perf, anchor_sizes[-1])
                print(opt_ext)
                print(best_so_far)
                # Stop evaluation if extrapolated performance is worse than the best seen so far
                if opt_ext >= best_so_far:                    
                    break

        return anchor_sequence

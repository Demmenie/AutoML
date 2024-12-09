import logging
import numpy as np
import pandas as pd
import scipy.stats as st
import typing

from vertical_model_evaluator import VerticalModelEvaluator

class LCCV(VerticalModelEvaluator):
    
    @staticmethod
    def optimistic_extrapolation(
        previous_anchor: int, previous_performance: float, 
        current_anchor: int, current_performance: float, target_anchor: int
    ) -> float:
        """
        Does the optimistic performance. Since we are working with a simplified
        surrogate model, we can not measure the infimum and supremum of the
        distribution. Just calculate the slope between the points, and
        extrapolate this.

        :param previous_anchor: See name
        :param previous_performance: Performance at previous anchor
        :param current_anchor: See name
        :param current_performance: Performance at current anchor
        :param target_anchor: the anchor at which we want to have the
        optimistic extrapolation
        :return: The optimistic extrapolation of the performance
        """
        
        # Mean of the performance values
        mean = (previous_performance + current_performance) / 2
        
        # Standard deviation
        sd = np.std([previous_performance, current_performance])

        # Making sure that we don't multiply by zero
        if sd == 0.0:
            sd = 0.00000000000000000001
        
        # sd = (abs(current_performance - mean)**2 +
        #       abs(previous_performance - mean)**2) / 2

        # Confidence intervals for the performance
        C_i = st.norm.interval([previous_performance, current_performance],
                               loc=mean,
                               scale=sd)
        # C_i = (np.pi * sd) * np.exp(-0.5*((-mean)/sd)**2)

        # supC_t1 = previous_performance + C_i
        # infC_t = current_performance - C_i
        
        # LCCV calculation: infimum of Ct - (sT - st)(supremum of Ct-1 - infimum of Ct /
        # st-1 - st)
        sub = ((C_i[0][1] - C_i[1][0]) / (previous_anchor - current_anchor))
        optPerf = C_i[1][0] - (target_anchor - current_anchor) * sub
        
        # print("C_i:", C_i)
        # print("sd:", sd)
        # print(optPerf)
        return optPerf
    

    def evaluate_model(self, best_so_far: typing.Optional[float],
                       configuration: typing.Dict) -> typing.List[typing.Tuple[int, float]]:
        """
        Does a staged evaluation of the model, on increasing anchor sizes.
        Determines after the evaluation at every anchor an optimistic
        extrapolation. In case the optimistic extrapolation can not improve
        over the best so far, it stops the evaluation.
        In case the best so far is not determined (None), it evaluates
        immediately on the final anchor (determined by self.final_anchor)

        :param best_so_far: indicates which performance has been obtained so far
        :param configuration: A dictionary indicating the configuration

        :return: A tuple of the evaluations that have been done. Each element of
        the tuple consists of two elements: the anchor size and the estimated
        performance.
        """

        anchorSize = 16
        anchorSequence = []
        x = pd.DataFrame(configuration, index=[0])

        if best_so_far == None:
            x["anchor_size"] = self.final_anchor
            best_so_far = self.surrogate_model.predict(x)
        

        while anchorSize < self.final_anchor:

            if anchorSize > (self.final_anchor * 0.75):
                anchorSize = self.final_anchor

            x["anchor_size"] = anchorSize
            predPerf = self.surrogate_model.predict(x)

            if len(anchorSequence) > 0:
                optExt = self.optimistic_extrapolation(anchorSequence[-1][0],
                                                       anchorSequence[-1][1],
                                                       anchorSize, predPerf,
                                                       self.final_anchor)
                
                #print(optExt, best_so_far)
                if optExt >= best_so_far:
                    break

            anchorSequence.append((anchorSize, predPerf))
            anchorSize = anchorSize * 2

        #print(anchorSequence)
        return anchorSequence
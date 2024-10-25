import ConfigSpace
import typing
import numpy as np
import pandas as pd
from scipy.stats import norm

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer



class SequentialModelBasedOptimization(object):
    def __init__(self):
        """
        Initializes empty variables for the model, the list of runs (capital R), and the incumbent
        (theta_inc being the best found hyperparameters, theta_inc_performance being the performance
        associated with it)
        """
        self.R = []
        self.theta = []
        self.theta_inc = None
        self.theta_inc_performance = None
        self.model_pipeline = None
        self.hp = ['metric', 'pp@cat_encoder', 'pp@decomposition', 'pp@featuregen',
                           'pp@featureselector', 'pp@scaler', 'weights', 'pp@kernel_pca_kernel', 'n_neighbors', 'p', 'pp@kernel_pca_n_components', 'pp@poly_degree',
                         'pp@selectp_percentile', 'pp@std_with_std', 'anchor_size']
        self.categorical = ['metric', 'pp@cat_encoder', 'pp@decomposition', 'pp@featuregen',
                           'pp@featureselector', 'pp@scaler', 'weights', 'pp@kernel_pca_kernel']

    def initialize(self, capital_phi: typing.List[typing.Tuple[typing.Dict, float]]) -> None:
        """
        Initializes the model with a set of initial configurations, before it can make recommendations
        which configurations are in good regions. Note that we are minimising (lower values are preferred)

        :param capital_phi: a list of tuples, each tuple being a configuration and the performance (typically,
        error rate)
        """        
        self.R = capital_phi.copy()

        best_run = min(self.R, key=lambda x: x[1])
        self.theta_inc = best_run[0]  
        self.theta_inc_performance = best_run[1]  
        
    def fit_model(self):   
        """
        Fits the internal surrogate model on the complete run list (self.R).
        """     
        y = np.array([])

        configs = [config[0] for config in self.R]
        X = self.process_config_structure(configs)
        for elem in self.R:            
            y = np.append(y, elem[1])
        
        self.theta = X

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), list(set(self.hp) - set(self.categorical))),
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical)
            ])

        self.model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('gpr', GaussianProcessRegressor(kernel=Matern(), normalize_y=True))
        ])

        self.model_pipeline.fit(X, y)
    
    def select_configuration(self) -> ConfigSpace.Configuration:
        """
        Determines which configurations are good, based on the internal surrogate model.
        Note that we are minimizing the error, but the expected improvement takes into account that.
        Therefore, we are maximizing expected improvement here.

        :return: The configuration with the highest expected improvement
        """
        self.theta = self.theta.to_dict(orient='records')
        self.theta = self.process_config_structure(self.theta)
        ei_list = self.expected_improvement(self.model_pipeline, self.theta_inc_performance, self.theta)
        best_index = np.argmax(ei_list) 

        best_config = self.theta.iloc[best_index].to_dict()

        return best_config
    
    @staticmethod
    def expected_improvement(model_pipeline: Pipeline, f_star: float, theta: np.array) -> np.array:
        """
        Acquisition function that determines which configurations are good and which
        are not good.

        :param model_pipeline: The internal surrogate model (should be fitted already)
        :param f_star: The current incumbent (theta_inc)
        :param theta: A (n, m) array, each column represents a hyperparameter and each row
        represents a configuration
        :return: A size n vector, same size as each element representing the EI of a given
        configuration
        """
        mu, sigma = model_pipeline.predict(theta, return_std=True)
        temp = (f_star-mu)
        ei = temp * norm.cdf(temp / sigma) + sigma*norm.pdf(temp / sigma)
        
        print(f"ei: {ei}")
        return ei

    def update_runs(self, run: typing.Tuple[typing.Dict, float]):
        """
        After a configuration has been selected and ran, it will be added to the run list
        (so that the model can be trained on it during the next iterations).
        The functions updates the incumbent configuration if the new configuration performs better.

        :param run: A tuple (configuration, performance) where performance is error rate.
        """
        self.R.append(run)
        if run[1] < self.theta_inc_performance:            
            self.theta_inc = run[0]
            self.theta_inc_performance = run[1]

    def process_config_structure(self, configs):
        """
        Processes the list of configurations by separating numerical and categorical features.
        It prepares them for use in the Gaussian Process model.

        :param configs: A list of configurations dictionaries where each configuration is a dictionary
        mapping hyperparameters to their values.
        :return: A pandas DataFrame with processed numerical and categorical features.
        """
        value_list_cat = []
        value_list_num = []
        for elem in configs:
            values_cat = [elem[key] if key in elem else 'none' for key in self.categorical]
            value_list_cat.append(values_cat)      
            values_num = [elem[key] if key in elem else np.nan for key in list(set(self.hp) - set(self.categorical)) ]
            value_list_num.append(values_num)
            
        df_num = pd.DataFrame(value_list_num, columns=list(set(self.hp) - set(self.categorical)))
        df_cat = pd.DataFrame(value_list_cat, columns=self.categorical)
        df_num.fillna(0, inplace=True)  
        df_cat.fillna('none', inplace=True) 
        
        df_full = pd.concat([df_num, df_cat], axis=1)
        
        return df_full
import ConfigSpace

import sklearn.impute
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

class SurrogateModel:

    def __init__(self, config_space):
        self.config_space = config_space
        self.df = None
        self.model = None
        self.encoder = LabelEncoder()

    def fit(self, df):
        """
        Receives a data frame, in which each column (except for the last two) represents a hyperparameter, the
        penultimate column represents the anchor size, and the final column represents the performance.

        :param df: the dataframe with performances
        :return: Does not return anything, but stores the trained model in self.model
        """
        
        self.model = RandomForestRegressor(n_estimators=15, max_features=15)
        self.df = df

        x = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        xCatFit = ["minkowski", "cosine", "nan_euclidean",
                   "onehot", "ordinal", "none", "kernel_pca", "lda", "fastica",
                   "ka_rbf", "ka_nystroem", "agglomerator", "poly", "selectp", 
                   "minmax", "std", "uniform", "distance", "linear", "rbf", "nan"]
        
        self.encoder.fit(xCatFit)
        
        xCat = pd.concat([pd.Series(x["metric"]),
            x.loc(axis=1)["pp@cat_encoder",
                "pp@decomposition",
                "pp@featuregen",
                "pp@featureselector",
                "pp@scaler",
                "weights"],
            pd.Series(x["pp@kernel_pca_kernel"])],
            axis=1)
        xCat = xCat.apply(self.encoder.transform)

        xNum = pd.concat([pd.Series(x["n_neighbors"]),
            pd.Series(x["p"]),
            x.loc(axis=1)["pp@kernel_pca_n_components",
                "pp@poly_degree",
                "pp@selectp_percentile",
                "pp@std_with_std",
                "anchor_size"]],
            axis=1)
        x = pd.concat([pd.DataFrame(xNum), pd.DataFrame(xCat)], axis=1).values


        xTrain, xVal, yTrain, yVal = train_test_split(x, y, test_size=0.3)

        self.model.fit(xTrain, yTrain)
        pred = self.model.predict(xVal)

        mse = mean_squared_error(yVal, pred)
        print(f'Mean Squared Error (val): {mse}')

        r2 = r2_score(yVal, pred)
        print(f'R-squared (val): {r2}')

        self.model.fit(x, y)


    def predict(self, theta_new):
        """
        Predicts the performance of a given configuration theta_new

        :param theta_new: a dict, where each key represents the hyperparameter (or anchor)
        :return: float, the predicted performance of theta new (which can be considered the ground truth)
        """
        
        x = pd.DataFrame(theta_new, columns=self.df.columns[:-1], index=[0])

        xCat = pd.concat([pd.Series(x["metric"]),
            x.loc(axis=1)["pp@cat_encoder",
                "pp@decomposition",
                "pp@featuregen",
                "pp@featureselector",
                "pp@scaler",
                "weights"],
            pd.Series(x["pp@kernel_pca_kernel"])],
            axis=1)
        xCat = xCat.apply(self.encoder.transform)

        xNum = pd.concat([pd.Series(x["n_neighbors"]),
            pd.Series(x["p"]),
            x.loc(axis=1)["pp@kernel_pca_n_components",
                "pp@poly_degree",
                "pp@selectp_percentile",
                "pp@std_with_std",
                "anchor_size"]],
            axis=1)
        x = pd.concat([pd.DataFrame(xNum), pd.DataFrame(xCat)], axis=1).values

        x = np.array(x)

        pred = self.model.predict(x)

        return pred[0]

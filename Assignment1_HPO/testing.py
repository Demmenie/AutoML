import argparse
import ConfigSpace
import matplotlib.pyplot as plt
import pandas as pd
from random_search import RandomSearch
from surrogate_model import SurrogateModel
from sklearn.metrics import mean_squared_error, r2_score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_space_file', type=str, default='lcdb_config_space_knn.json')
    parser.add_argument('--configurations_performance_file', type=str, default='lcdb_configs.csv')
    # max_anchor_size: connected to the configurations_performance_file. The max value upon which anchors are sampled
    parser.add_argument('--max_anchor_size', type=int, default=1600)
    parser.add_argument('--num_iterations', type=int, default=5000)

    return parser.parse_args()


def run(args):
    config_space = ConfigSpace.ConfigurationSpace.from_json(args.config_space_file)
    random_search = RandomSearch(config_space)
    df = pd.read_csv(args.configurations_performance_file)
    surrogate_model = SurrogateModel(config_space)

    surrogate_model.fit(df.iloc[:1200])

    df = df.to_numpy()

    predList = []
    for x in df[1200:, :-1]:
        x = {'metric': x[0],
             'n_neighbors': x[1],
             'pp@cat_encoder': x[2],
             'pp@decomposition': x[3],
             'pp@featuregen': x[4],
             'pp@featureselector': x[5],
             'pp@scaler': x[6],
             'weights': x[7],
             'anchor_size': x[8]}
        pred = surrogate_model.predict(x)
        predList.append(pred)

    mse = mean_squared_error(df[1200:, -1], predList)
    print(f'Mean Squared Error (test): {mse}')

    r2 = r2_score(df[1200:, -1], predList)
    print(f'R-squared (test): {r2}')




run(parse_args())
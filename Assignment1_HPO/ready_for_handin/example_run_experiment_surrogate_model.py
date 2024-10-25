import argparse
import ConfigSpace
import matplotlib.pyplot as plt
import pandas as pd
from random_search import RandomSearch
from surrogate_model import SurrogateModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_space_file', type=str, default='lcdb_config_space_knn.json')
    parser.add_argument('--configurations_performance_file', type=str, default='config-performances\lcdb_configs.csv')
    # max_anchor_size: connected to the configurations_performance_file. The max value upon which anchors are sampled
    parser.add_argument('--max_anchor_size', type=int, default=1600)
    parser.add_argument('--num_iterations', type=int, default=500)

    return parser.parse_args()


def run_surrogate_model(args):
    """
    Executes a Random Search using a surrogate model to predict performance, tracking the best performance over iterations.

    Args:
        args (argparse.Namespace): Command-line arguments parsed via argparse.
        include_plot (bool): If True, plots performance over iterations.
        set_seed (bool): If True, sets the random seed for reproducibility.
        seed (int): Random seed value.

    Returns:
        list: A list of best performance values over the iterations.
    """
    config_space = ConfigSpace.ConfigurationSpace.from_json(args.config_space_file)
    random_search = RandomSearch(config_space)
    df = pd.read_csv(args.configurations_performance_file)
    surrogate_model = SurrogateModel(config_space)
    surrogate_model.fit(df)
    results = {
        'random_search': [0.2]
    }

    for _ in range(args.num_iterations):
        theta_new = dict(random_search.select_configuration())
        theta_new['anchor_size'] = args.max_anchor_size
        performance = surrogate_model.predict(theta_new)
        # ensure to only record improvements
        results['random_search'].append(min(results['random_search'][-1], performance))
        random_search.update_runs((theta_new, performance))

    plt.plot(range(len(results['random_search'])), results['random_search'])
    plt.yscale('log')
    plt.show()


if __name__ == '__main__':
    run_surrogate_model(parse_args())

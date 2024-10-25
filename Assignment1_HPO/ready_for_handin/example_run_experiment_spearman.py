import argparse
import ConfigSpace
import matplotlib.pyplot as plt
import pandas as pd
from random_search import RandomSearch
from surrogate_model import SurrogateModel
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_space_file', type=str, default='lcdb_config_space_knn.json')
    parser.add_argument('--configurations_performance_file', type=str, default='config-performances\lcdb_configs.csv')
    # max_anchor_size: connected to the configurations_performance_file. The max value upon which anchors are sampled
    parser.add_argument('--max_anchor_size', type=int, default=1600)
    parser.add_argument('--num_iterations', type=int, default=5000)

    return parser.parse_args()


def run_spearman(args):
    """
    Runs random search optimization with a surrogate model and calculates the Spearman correlation 
    between predicted and actual configuration performances on a test set. It also tracks and plots 
    the error over the specified number of iterations.

    Args:
        args (argparse.Namespace): Command-line arguments with:
            - config_space_file (str): Path to the configuration space JSON.
            - configurations_performance_file (str): Path to the CSV file with configuration performances.
            - max_anchor_size (int): Maximum anchor size.
            - num_iterations (int): Number of iterations for the random search.

    Returns:
        None
    """
    config_space = ConfigSpace.ConfigurationSpace.from_json(args.config_space_file)
    random_search = RandomSearch(config_space)
    df = pd.read_csv(args.configurations_performance_file)

    train, test = train_test_split(df)

    surrogate_model = SurrogateModel(config_space)
    surrogate_model.fit(train)
    results = {
        'random_search': [0.4]
    }

    total = 0
    count = 0
    for row in test.iterrows():
        count += 1
        row = pd.DataFrame(row).iloc[-1].iloc[-1]
    
        act = row.loc["score"]
        pred = surrogate_model.predict(row)

        diff = act - pred
        total += (diff**2)

    spearman = (6*total) / (count * (count**2 - 1))


    for _ in range(args.num_iterations):
        theta_new = dict(random_search.select_configuration())
        theta_new['anchor_size'] = args.max_anchor_size
        performance = surrogate_model.predict(theta_new)
        # ensure to only record improvements
        results['random_search'].append(float(min(results['random_search'][-1], performance)))
        random_search.update_runs((theta_new, performance))
       
    
    print("Spearman Correlation:", spearman)
    plt.plot(range(len(results['random_search'])), results['random_search'])
    plt.ylabel('Error')
    plt.xlabel('Iterations')
    plt.yscale('log')
    plt.show()


if __name__ == '__main__':
    run_spearman(parse_args())
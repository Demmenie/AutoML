import argparse
import os
import ConfigSpace as CS
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from random_search import RandomSearch
from surrogate_model import SurrogateModel
from succesive_halving import SuccesiveHalving
from smbo import SequentialModelBasedOptimization


def df_to_typed_list(df: pd.DataFrame) -> list:
    """
    Converts a DataFrame of configurations and performances into a list of tuples.
    Each tuple contains a dictionary of configuration parameters and the associated performance score.

    Args:
        df (pd.DataFrame): DataFrame containing configurations as columns and performance scores as the last column.

    Returns:
        list: A list of tuples with configuration dictionaries and their associated performance scores.
    """
    score_column = df.columns[-1]
    
    result = [
        (row.drop(score_column).to_dict(), row[score_column]) for _, row in df.iterrows()
    ]
    
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_space_file', type=str, default='lcdb_config_space_knn.json')
    parser.add_argument('--configurations_performance_file', type=str, default='config-performances/lcdb_configs.csv')
    parser.add_argument('--max_anchor_size', type=int, default=1600)
    parser.add_argument('--num_iterations', type=int, default=5000)

    return parser.parse_args()


def generate_configs(config_space, surrogate_model, with_performance, num_configs=10):
    """
    Generates a set of configurations from the configuration space and optionally predicts their performance.

    Args:
        config_space (CS.ConfigurationSpace): The configuration space to sample from.
        surrogate_model (SurrogateModel): A surrogate model to predict performance.
        with_performance (bool): If True, predicts and appends performance values to the configurations.
        num_configs (int): Number of configurations to generate.

    Returns:
        list: List of generated configurations. Each configuration is either a dictionary or 
              a tuple of (configuration dictionary, predicted performance).
    """
    configs = []
    
    for _ in range(num_configs):
        config = dict(config_space.sample_configuration())  
        if with_performance:     
            performance = surrogate_model.predict(config)               
            configs.append((config, performance))    
        else:
            configs.append((config))  
    return configs


def run_surrogate_model(args, include_plot=True, set_seed=True, seed=None):
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
    if set_seed and seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    config_space = CS.ConfigurationSpace.from_json(args.config_space_file)
    random_search = RandomSearch(config_space)
    df = pd.read_csv(args.configurations_performance_file)

    surrogate_model = SurrogateModel(config_space)
    surrogate_model.fit(df)
    results = {
        'random_search': [0.16]  
    }

    for _ in range(500):
        theta_new = dict(random_search.select_configuration())
        theta_new['anchor_size'] = args.max_anchor_size
        performance = surrogate_model.predict(theta_new)
        # ensure to only record improvements
        results['random_search'].append(min(results['random_search'][-1], performance))
        random_search.update_runs((theta_new, performance))

    if include_plot:
        plt.plot(range(len(results['random_search'])), results['random_search'])
        plt.yscale('log')
        plt.show()

    return results['random_search']


def run_smbo(args, include_plot=True, set_seed=True, seed=None):
    """
    Executes Sequential Model-Based Optimization (SMBO) and tracks the best performance over iterations.

    Args:
        args (argparse.Namespace): Command-line arguments parsed via argparse.
        include_plot (bool): If True, plots performance over iterations.
        set_seed (bool): If True, sets the random seed for reproducibility.
        seed (int): Random seed value.

    Returns:
        list: A list of best performance values over the iterations.
    """
    if set_seed and seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    config_space = CS.ConfigurationSpace.from_json(args.config_space_file)
    df = pd.read_csv(args.configurations_performance_file)

    surrogate_model_external = SurrogateModel(config_space)
    surrogate_model_external.fit(df)

    capital_phi = generate_configs(config_space, surrogate_model_external, with_performance=True, num_configs=30)

    surrogate_model_internal = SequentialModelBasedOptimization() 
    surrogate_model_internal.initialize(capital_phi)
    error_list = []

    for _ in range(300):
        surrogate_model_internal.fit_model()
        sample_config = config_space.sample_configuration(size=10)

        if not isinstance(sample_config, list):
            sample_config = [sample_config]    

        sample_config = pd.DataFrame(sample_config)
        sample_config['anchor_size'] = args.max_anchor_size
        
        surrogate_model_internal.theta = sample_config
        
        theta_new = dict(surrogate_model_internal.select_configuration())
        theta_new['anchor_size'] = args.max_anchor_size
        performance = surrogate_model_external.predict(theta_new)
        
        surrogate_model_internal.update_runs((theta_new, performance))
        result = float(surrogate_model_internal.theta_inc_performance)
        error_list.append(result)

    if include_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(error_list)), error_list, label='Best Performance Over Time', color='blue')
        plt.xlabel('Iteration')
        plt.ylabel('Best Performance')
        plt.title('Best Performance Over Iterations')
        plt.grid(True)
        plt.show()

    return error_list


def run_successive_halving(args, include_plot=True, include_best_performance=True):
    """
    Runs Successive Halving algorithm and tracks the best performance over time.

    Args:
        args (argparse.Namespace): Command-line arguments parsed via argparse.
        include_plot (bool): If True, plots performance over iterations.
        include_best_performance (bool): If True, returns the best performance found during the run.

    Returns:
        list or None: A list containing the best performance if `include_best_performance` is True, otherwise None.
    """
    dataset_dir = "config-performances"
    
    datasets = [f for f in os.listdir(dataset_dir) if f.endswith('.csv')]

    for dataset in datasets:
        config_space = CS.ConfigurationSpace.from_json(args.config_space_file)
        df = pd.read_csv(os.path.join(dataset_dir, dataset))

        surrogate_model_external = SurrogateModel(config_space)
        surrogate_model_external.fit(df)

        initial_anchor_size = 25
        num_initial_samples = 64
        max_anchor_size = args.max_anchor_size
        reduction_factor = 2
        succesive_halving = SuccesiveHalving(surrogate_model_external, initial_anchor_size, max_anchor_size, reduction_factor)

        sample_configs = config_space.sample_configuration(size=num_initial_samples)
        succesive_halving.config_queue = sample_configs
        succesive_halving.initialize()
        succesive_halving.update_queue(sample_configs)

        while succesive_halving.anchor_size <= max_anchor_size:    
            best_configs = succesive_halving.select_best_configurations()
            succesive_halving.update_queue(best_configs)

        if include_plot:
            succesive_halving.plot()
        
        if include_best_performance:
            filtered_configs = {k: v for k, v in succesive_halving.performance_history.items() if len(v) == 7}
            if filtered_configs:
                best_config = min(filtered_configs, key=lambda k: filtered_configs[k][-1])
                best_performance = filtered_configs[best_config]

                return best_performance  


def compare_sm_smbo_over_runs(args, num_runs=10):
    """
    Compare the average performance of Random Search and SMBO over multiple runs.

    Args:
        args (argparse.Namespace): Command-line arguments parsed via argparse.
        num_runs (int): Number of independent runs to average the results.

    Returns:
        None
    """
    random_search_all_performances = []
    smbo_all_performances = []
    
    for _ in range(num_runs):
        random_search_performance = run_surrogate_model(args, False, True, 2024)    
        smbo_performance = run_smbo(args, False, True, 2024)
        
        min_length = min(len(random_search_performance), len(smbo_performance))
        random_search_performance = random_search_performance[:min_length]
        smbo_performance = smbo_performance[:min_length]
        
        random_search_all_performances.append(random_search_performance)
        smbo_all_performances.append(smbo_performance)
    
    avg_random_search_performance = np.mean(random_search_all_performances, axis=0)
    avg_smbo_performance = np.mean(smbo_all_performances, axis=0)
    
    iterations = range(len(avg_random_search_performance))
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(iterations, avg_random_search_performance, label='Average Random Search')
    plt.plot(iterations, avg_smbo_performance, label='Average SMBO')
    
    plt.xlabel('Iteration')
    plt.ylabel('Best Score')
    plt.title(f'Average Score Comparison over {num_runs} Runs: Random Search vs. SMBO')
    plt.yscale('log') 
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':    
    run_surrogate_model(parse_args(), True, True, 2024)
    run_smbo(parse_args(), True, True, 2024)
    run_successive_halving(parse_args(), True, False)
    compare_sm_smbo_over_runs(parse_args(), num_runs=5)

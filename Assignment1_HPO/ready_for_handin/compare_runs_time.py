import argparse
import ConfigSpace as CS
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import time
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
    parser.add_argument('--configurations_performance_file', type=str, default='config-performances\lcdb_configs.csv')
    # max_anchor_size: connected to the configurations_performance_file. The max value upon which anchors are sampled
    parser.add_argument('--max_anchor_size', type=int, default=1600)
    parser.add_argument('--num_iterations', type=int, default=5000)

    return parser.parse_args()


def generate_configs(config_space, surrogate_model, with_performance, num_configs=10):
    """
    Generate configurations from the configuration space and optionally predict their performance.

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


def run_surrogate_model(args, include_plot=True, set_seed=True, seed=None, max_seconds=10):
    """
    Executes Random Search with a surrogate model to predict performance over time.

    Args:
        args (argparse.Namespace): Command-line arguments parsed via argparse.
        include_plot (bool): If True, plots performance over time.
        set_seed (bool): If True, sets the random seed for reproducibility.
        seed (int): Random seed value.
        max_seconds (int): Maximum time to run the search in seconds.

    Returns:
        tuple: Contains a list of time intervals and the corresponding performance values at each interval.
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

    start_time = time.time()
    performances_over_time = []
    times = []

    while time.time() - start_time < max_seconds:  
        elapsed_time = time.time() - start_time
        theta_new = dict(random_search.select_configuration())
        theta_new['anchor_size'] = args.max_anchor_size
        performance = surrogate_model.predict(theta_new)

        results['random_search'].append(min(results['random_search'][-1], performance))
        random_search.update_runs((theta_new, performance))

        performances_over_time.append(results['random_search'][-1])
        times.append(elapsed_time)

    if include_plot:
        plt.plot(times, performances_over_time, label='Random Search')
        plt.yscale('log')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Performance')
        plt.title('Random Search Performance Over Time')
        plt.grid(True)
        plt.legend()
        plt.show()

    return times, performances_over_time


def run_smbo(args, include_plot=True, set_seed=True, seed=None, max_seconds=10):
    """
    Executes Sequential Model-Based Optimization (SMBO) over time.

    Args:
        args (argparse.Namespace): Command-line arguments parsed via argparse.
        include_plot (bool): If True, plots performance over time.
        set_seed (bool): If True, sets the random seed for reproducibility.
        seed (int): Random seed value.
        max_seconds (int): Maximum time to run SMBO in seconds.

    Returns:
        tuple: Contains a list of time intervals and the corresponding performance values at each interval.
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

    start_time = time.time()
    performances_over_time = []
    times = []

    while time.time() - start_time < max_seconds: 
        elapsed_time = time.time() - start_time
        surrogate_model_internal.fit_model()
        
        sample_config = config_space.sample_configuration(size=10)  
        sample_config = pd.DataFrame(sample_config)
        sample_config['anchor_size'] = args.max_anchor_size
        
        surrogate_model_internal.theta = sample_config
        
        theta_new = dict(surrogate_model_internal.select_configuration())
        theta_new['anchor_size'] = args.max_anchor_size
        performance = surrogate_model_external.predict(theta_new)
        
        surrogate_model_internal.update_runs((theta_new, performance))
        result = float(surrogate_model_internal.theta_inc_performance)
        error_list.append(result)

        performances_over_time.append(result)
        times.append(elapsed_time)

    if include_plot:
        plt.plot(times, performances_over_time, label='SMBO')
        plt.yscale('log')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Performance')
        plt.title('SMBO Performance Over Time')
        plt.grid(True)
        plt.legend()
        plt.show()

    return times, performances_over_time


def run_successive_halving(args, include_plot=True):
    """
    Executes Successive Halving algorithm over time and tracks the best performance.

    Args:
        args (argparse.Namespace): Command-line arguments parsed via argparse.
        include_plot (bool): If True, plots performance over time.

    Returns:
        tuple: Contains a list of time intervals and the corresponding best performance values.
    """
    config_space = CS.ConfigurationSpace.from_json(args.config_space_file)
    df = pd.read_csv(args.configurations_performance_file)

    surrogate_model_external = SurrogateModel(config_space)
    surrogate_model_external.fit(df)

    initial_anchor_size = 100
    num_initial_samples = 64
    max_anchor_size = args.max_anchor_size
    reduction_factor = 2

    performances_over_time = []
    global_best_performance = np.inf
    times = []

    start_time = time.time()

    while time.time() - start_time < 180:
        succesive_halving = SuccesiveHalving(surrogate_model_external, initial_anchor_size, max_anchor_size, reduction_factor)
        
        elapsed_time = time.time() - start_time
        sample_configs = config_space.sample_configuration(size=num_initial_samples)
        
        succesive_halving.config_queue = sample_configs
        succesive_halving.initialize()
        succesive_halving.update_queue(sample_configs)
        
        filtered_configs = {k: v for k, v in succesive_halving.performance_history.items()}
        best_config = min(filtered_configs, key=lambda k: filtered_configs[k][-1])
        best_performance = filtered_configs[best_config][0]
        
        if best_performance < global_best_performance:
            global_best_performance = best_performance

        performances_over_time.append(global_best_performance)
        times.append(elapsed_time)

        while succesive_halving.anchor_size <= max_anchor_size and time.time() - start_time < 30:
            elapsed_time = time.time() - start_time

            best_configs = succesive_halving.select_best_configurations()
            succesive_halving.update_queue(best_configs)

            filtered_configs = {k: v for k, v in succesive_halving.performance_history.items()}
            best_config = min(filtered_configs, key=lambda k: filtered_configs[k][-1])
            best_performance = filtered_configs[best_config][-1]

            if best_performance < global_best_performance:
                global_best_performance = best_performance

            performances_over_time.append(global_best_performance)
            times.append(elapsed_time)

    if include_plot:
        plt.plot(times, performances_over_time, label='Successive Halving')
        plt.yscale('log')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Best Performance')
        plt.title('Successive Halving Performance Over Time')
        plt.grid(True)
        plt.legend()
        plt.show()

    return times, performances_over_time


def run_all_methods(args, set_seed=True, seed=None, num_runs=10, num_time_intervals=100):
    """
    Runs Random Search, SMBO, and Successive Halving methods, and compares their performance over time.

    Args:
        args (argparse.Namespace): Command-line arguments parsed via argparse.
        set_seed (bool): If True, sets the random seed for reproducibility.
        seed (int): Random seed value.
        num_runs (int): Number of independent runs to average the results.
        num_time_intervals (int): Number of time intervals for result interpolation.

    Returns:
        None
    """
    successive_halving_results_all = []
    surrogate_results_all = []
    smbo_results_all = []
    
    max_time = 0
    
    for run in range(num_runs):
        print(f"Run {run+1}/{num_runs}")
        
        # Run Successive Halving
        times_successive, _ = run_successive_halving(args, include_plot=False)
        max_time = max(max_time, times_successive[-1])

        # Run Surrogate Model
        times_surrogate, _ = run_surrogate_model(args, include_plot=False, set_seed=set_seed, seed=seed, max_seconds=times_successive[-1])
        max_time = max(max_time, times_surrogate[-1])

        # Run SMBO
        times_smbo, _ = run_smbo(args, include_plot=False, set_seed=set_seed, seed=seed, max_seconds=times_successive[-1])
        max_time = max(max_time, times_smbo[-1])

    common_times = np.linspace(0, max_time, num_time_intervals)

    for run in range(num_runs):
        print(f"Run {run+1}/{num_runs} (interpolating results)")
        
        # Run Successive Halving
        times_successive, successive_halving_results = run_successive_halving(args, include_plot=False)
        interpolated_successive_halving = np.interp(common_times, times_successive, successive_halving_results)
        successive_halving_results_all.append(interpolated_successive_halving)

        # Run Surrogate Model
        times_surrogate, surrogate_results = run_surrogate_model(args, include_plot=False, set_seed=set_seed, seed=seed, max_seconds=times_successive[-1])
        interpolated_surrogate = np.interp(common_times, times_surrogate, surrogate_results)
        surrogate_results_all.append(interpolated_surrogate)

        # Run SMBO
        times_smbo, smbo_results = run_smbo(args, include_plot=False, set_seed=set_seed, seed=seed, max_seconds=times_successive[-1])
        interpolated_smbo = np.interp(common_times, times_smbo, smbo_results)
        smbo_results_all.append(interpolated_smbo)

    successive_halving_results_all = np.array(successive_halving_results_all)
    surrogate_results_all = np.array(surrogate_results_all)
    smbo_results_all = np.array(smbo_results_all)

    avg_successive_halving_results = np.mean(successive_halving_results_all, axis=0)
    avg_surrogate_results = np.mean(surrogate_results_all, axis=0)
    avg_smbo_results = np.mean(smbo_results_all, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(common_times, avg_surrogate_results, label='Average Random Search')
    plt.plot(common_times, avg_smbo_results, label='Average SMBO')
    plt.plot(common_times, avg_successive_halving_results, label='Average Successive Halving')
    
    plt.yscale('log')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Best Score')
    plt.title('Average Score Comparison over 10 Runs: Random Search vs. SMBO vs. Successive Halving')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == '__main__': 
    run_all_methods(parse_args(), set_seed=True, seed=2024, num_runs=10)

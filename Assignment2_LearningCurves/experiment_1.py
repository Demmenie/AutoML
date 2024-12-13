import argparse
import ConfigSpace
import logging
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from lccv import LCCV
from surrogate_model import SurrogateModel
from IPL import IPL

def parse_args(dataset):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_space_file', type=str, default='lcdb_config_space_knn.json')
    parser.add_argument('--configurations_performance_file', type=str, default=f'config_performances_dataset-{dataset}.csv')
    parser.add_argument('--minimal_anchor', type=int, default=16)
    parser.add_argument('--max_anchor_size', type=int, default=16000)
    parser.add_argument('--num_iterations', type=int, default=300)
    return parser.parse_args()


def run_lccv(args, anchor_sizes):
    config_space = ConfigSpace.ConfigurationSpace.from_json(args.config_space_file)
    df = pd.read_csv(args.configurations_performance_file)
    surrogate_model = SurrogateModel(config_space)
    surrogate_model.fit(df)
    lccv = LCCV(surrogate_model=surrogate_model, minimal_anchor=anchor_sizes[0], final_anchor=anchor_sizes[-1])

    best_so_far = float('inf')  # Initialize the best performance as infinity
    # cumulative_best_performance = []  # To track the best performance over evaluations

    for idx in range(args.num_iterations):
        theta_new = dict(config_space.sample_configuration())
        result = lccv.evaluate_model(anchor_sizes, best_so_far, theta_new)
        for _, performance in result:
            best_so_far = min(best_so_far, performance)  # Update best performance
            # cumulative_best_performance.append(best_so_far)  # Track the best so far

    return lccv.cumulative_best_performance


def run_ipl(args, anchor_sizes, schedule_length):
    """Run the IPL evaluation and return cumulative best performance."""
    config_space = ConfigSpace.ConfigurationSpace.from_json(args.config_space_file)
    df = pd.read_csv(args.configurations_performance_file)
    surrogate_model = SurrogateModel(config_space)
    surrogate_model.fit(df)

    fixed_schedule = anchor_sizes[:schedule_length]
    final_max_anchor = anchor_sizes[-1]

    ipl = IPL(
        fixed_schedule=fixed_schedule,
        final_anchor=final_max_anchor,
        best_seen_performance=float('inf'),
    )

    best_so_far = float('inf')  # Initialize the best performance as infinity
    cumulative_best_performance = []  # To track the best performance over evaluations

    for idx in range(args.num_iterations):
        theta_new = dict(config_space.sample_configuration())
        result = []

        # Evaluate configuration using the fixed schedule
        for anchor_size in fixed_schedule:
            simulated_performance = surrogate_model.predict(theta_new, anchor_size)
            if best_so_far == float('inf'):
                best_so_far = simulated_performance            
                
            for _ in range(anchor_size):
                cumulative_best_performance.append(best_so_far)
            result.append((anchor_size, simulated_performance))

        y_values = [r[1] for r in result]  # Extract performance values
        if not ipl.evaluate_configuration(performances=y_values):
            continue  # Skip configurations that fail early stopping

        # Perform full evaluation for configurations that pass
        for anchor in anchor_sizes:
            real_performance = surrogate_model.predict(theta_new, anchor)
            best_so_far = min(best_so_far, real_performance)
        for _ in range(anchor_sizes[-1]):
            cumulative_best_performance.append(best_so_far)  # Track the cumulative best

    return cumulative_best_performance

def run_random_search(args, anchor_sizes):
    """Run Random Search and return cumulative best performance and evaluation counts."""
    config_space = ConfigSpace.ConfigurationSpace.from_json(args.config_space_file)
    df = pd.read_csv(args.configurations_performance_file)
    surrogate_model = SurrogateModel(config_space)
    surrogate_model.fit(df)

    best_so_far = None  # Initialize the best performance as infinity
    cumulative_best_performance = []  # To track the best performance over evaluations

    total_evals = 0  # Total evaluations counter

    for idx in range(args.num_iterations):
        # Sample a random configuration
        theta_new = dict(config_space.sample_configuration())

        # Predict performance for the largest anchor size
        random_anchor_size = int(random.uniform(16, 8098))
        predicted_performance = surrogate_model.predict(theta_new, random_anchor_size)
        total_evals += 1  # Increment total evaluations
        if not best_so_far:
            best_so_far = predicted_performance
        # Update the best performance seen so far
        best_so_far = min(best_so_far, predicted_performance)
        for _ in range(random_anchor_size):
            cumulative_best_performance.append(best_so_far)

    return cumulative_best_performance


def plot_comparison(lccv_best, ipl_best, rs_best, model_labels, save_path):
    """Plot the cumulative best performances for LCCV and IPL."""
    # epsilon = 1e-10
    # lccv_best = [max(x, epsilon) for x in lccv_best]
    # ipl_best = [max(x, epsilon) for x in ipl_best]
    # rs_best = [max(x, epsilon) for x in rs_best]

    plt.figure(figsize=(12, 8))
    # Plot LCCV cumulative best
    plt.plot(range(len(lccv_best)), lccv_best, label=model_labels[0], alpha=0.8)
    # Plot IPL cumulative best
    plt.plot(range(len(ipl_best)), ipl_best, label=model_labels[1], alpha=0.8)
    # Plot RS cumulative best
    plt.plot(range(len(rs_best)), rs_best, label=model_labels[2], alpha=0.8)

    plt.xlabel("Evaluations on Full Dataset")
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel("Performance")
    plt.title("Best Performance Over Evaluations")
    plt.xlim(1, len(rs_best)-3) 
    plt.ylim(0, 0.6)
    plt.yticks([0.1, 0.6], labels=[0.1, 0.6])
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  # Save with high resolution
    plt.show()


def pad_to_equal_length(runs):
    """
    Pads all runs to the same length using the last value of each run.
    
    :param runs: List of lists, where each sublist represents a run.
    :return: Numpy array with all runs padded to the same length.
    """
    max_length = max(len(run) for run in runs)
    padded_runs = []
    for run in runs:
        padded_runs.append(run + [run[-1]] * (max_length - len(run)))  # Pad with last value
    return np.array(padded_runs)

def every_nth_element(array, n):
    array.insert(0, 1)
    return array[::n] 

if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    DATASET = 6
    anchor_dict_LCCV = {6: [128, 256, 512, 1024, 2048, 4096, 8192, 16000],
                        11: [16, 32, 64, 128, 256, 512],
                        1457: [64, 128, 256, 512, 1024, 2048]} 
    anchor_dict_IPL = {6: [16, 32, 64, 128, 256, 512, 1024, 2048, 8192, 16000],
                        11: [16, 32, 64, 128, 256, 512, 1024, 2048, 8192, 16000],
                        1457: [16, 32, 64, 128, 256, 512, 1024, 2048, 8192, 16000]} 
    schedule_length = 4  
    anchor_sizes_LCCV = anchor_dict_LCCV[DATASET]
    num_runs = 10 # Number of runs to average over
    anchor_sizes_IPL = anchor_dict_IPL[DATASET]

    # Parse arguments
    args = parse_args(DATASET)

    # Collect cumulative best performances over multiple runs
    lccv_runs = []
    ipl_runs = []
    rs_runs = []

    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}")

        # Run LCCV
        lccv_best_performance = run_lccv(args, anchor_sizes_LCCV)        
        lccv_runs.append(every_nth_element(lccv_best_performance, anchor_sizes_LCCV[-1]))

        # Run IPL
        ipl_best_performance = run_ipl(args, anchor_sizes_IPL, schedule_length)
        ipl_runs.append(every_nth_element(ipl_best_performance, anchor_sizes_IPL[-1]))

        # Run RS
        rs_best_performance = run_random_search(args, anchor_sizes_IPL)
        rs_runs.append(every_nth_element(rs_best_performance, anchor_sizes_IPL[-1]))

    # Pad results to equal lengths
    lccv_runs_padded = pad_to_equal_length(lccv_runs)
    ipl_runs_padded = pad_to_equal_length(ipl_runs)
    rs_runs_padded = pad_to_equal_length(rs_runs)

    # Compute the average cumulative best performance
    lccv_avg = np.mean(lccv_runs_padded, axis=0)
    ipl_avg = np.mean(ipl_runs_padded, axis=0)
    rs_avg = np.mean(rs_runs_padded, axis=0)

    # Plot the averaged results
    plot_comparison(
        lccv_avg,
        ipl_avg,
        rs_avg,
        model_labels=["LCCV", f"IPL (Schedule Length {6})", "Random Search"],
        save_path="images\comparison_plot_evaluation.png"
    )
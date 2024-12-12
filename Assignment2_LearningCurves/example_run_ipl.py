import argparse
import ConfigSpace
import logging
import matplotlib.pyplot as plt
import pandas as pd
from lccv import LCCV
from surrogate_model import SurrogateModel
from IPL import IPL

def plot_results_schedules(results_dict, save_path):
    """
    Plot results for multiple schedule lengths.
    :param results_dict: Dictionary with schedule length as keys and results as values
    """
    plt.figure(figsize=(10, 6))

    # Plot each result
    for schedule_length, result in results_dict.items():
        plt.plot(range(len(result)), result, linestyle='-', label=f"Schedule Length {schedule_length}")

    # Label the axes
    plt.xlabel("Iteration")
    plt.ylabel("Performance")
    plt.title("Performance for Different Fixed Schedule Lengths over Iterations")

    # Add legend
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  # Save with high resolution

    # Show the plot
    plt.show()

def plot_run_logs(logs, discarded_logs, anchor_sizes):
    # Plot results
    plt.figure(figsize=(10, 6))
    
    # Plot discarded configurations
    for x_data, y_data in discarded_logs:
        plt.plot(x_data, y_data, "x--", alpha=0.4)
    
    # Plot fully evaluated configurations
    for best_configs_results in logs:
        x_values = [i[0] for i in best_configs_results]
        y_values = [i[1] for i in best_configs_results]
        plt.plot(x_values, y_values, "o-", alpha=0.8)

    # Custom legend
    plt.plot([], [], "x--", color='black', label="Discarded Config")  # Dotted line for discarded
    plt.plot([], [], "o-", color='black', label="Evaluated Config")   # Solid line for evaluated

    plt.xlabel("Anchor Size")    
    plt.xscale("log")
    plt.xticks(anchor_sizes, labels=anchor_sizes)
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Configurations: Evaluated vs Discarded")
    plt.show()

def parse_args(dataset):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_space_file', type=str, default='lcdb_config_space_knn.json')
    parser.add_argument('--configurations_performance_file', type=str, default=f'config_performances_dataset-{dataset}.csv')
    parser.add_argument('--minimal_anchor', type=int, default=256)
    parser.add_argument('--max_anchor_size', type=int, default=16000)
    parser.add_argument('--num_iterations', type=int, default=500)

    return parser.parse_args()

def run(args, anchor_sizes, schedule_length, plot_run=True): 
    config_space = ConfigSpace.ConfigurationSpace.from_json(args.config_space_file)
    
    df = pd.read_csv(args.configurations_performance_file)

    # Initialize surrogate model and fit it with the data
    surrogate_model = SurrogateModel(config_space)
    surrogate_model.fit(df)

    # Define the learning curve schedule
    fixed_schedule = anchor_sizes[:schedule_length]
    final_max_anchor = anchor_sizes[-1]
    
    # Initialize the IPLModelEvaluator
    evaluator = IPL(
        fixed_schedule=fixed_schedule,
        final_anchor=final_max_anchor,
        best_seen_performance=float('inf'),        
    )

    logs = []  # Track configurations that passed
    discarded_logs = []  # Track configurations that were discarded

    for idx in range(args.num_iterations):
        theta_new = dict(config_space.sample_configuration())
        result = []

        for anchor_size in fixed_schedule:
            simulated_performance = surrogate_model.predict(theta_new, anchor_size)
            result.append((anchor_size, simulated_performance))
        
        # Extract x (sizes) and y (losses) data for evaluation
        x_data = [r[0] for r in result]
        y_data = [r[1] for r in result]

        # Evaluate configuration using IPL evaluator
        if not evaluator.evaluate_configuration(performances=y_data):
            discarded_logs.append((x_data, y_data))  # Track discarded configurations
            continue

        # Perform full evaluation if passed early stopping
        print(f"Configuration {idx} passed early evaluation. Performing full evaluation.")
        best_configs_results = []
        for anchor in anchor_sizes:
            real_performance = surrogate_model.predict(theta_new, anchor)
            best_configs_results.append((anchor, real_performance))
        if real_performance < evaluator.best_seen_performance:
            evaluator.best_seen_performance = real_performance
        logs.append(best_configs_results)

        if plot_run:
            plot_run_logs(logs, discarded_logs, anchor_sizes)

    return evaluator.performance_over_iterations

if __name__ == "__main__":
    DATASET = 1147
    anchor_dict = {6: [16, 32, 64, 128, 256, 512, 1024, 2048, 8192, 16000],
                   11: [16, 32, 64, 128, 256, 512, 1024, 2048, 8192, 16000],
                   1147: [16, 32, 64, 128, 256, 512, 1024, 2048, 8192, 16000]}  
    
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Store results for different fixed schedule lengths
    schedule_lengths = [4, 5, 6, 7]  # List of schedule lengths to evaluate
    results_dict = {}

    for length in schedule_lengths:
        aggregated_results = []  # To store multiple runs
        for _ in range(15):  # Run each schedule length 10 times
            result = run(parse_args(DATASET), anchor_sizes=anchor_dict[DATASET], schedule_length=length, plot_run=False)
            aggregated_results.append(result)
        
        # Average results per iteration
        averaged_results = [sum(x) / len(x) for x in zip(*aggregated_results)]
        results_dict[length] = averaged_results

    # Plot the results
    plot_results_schedules(results_dict, save_path="images\schedule_length_plot.png")
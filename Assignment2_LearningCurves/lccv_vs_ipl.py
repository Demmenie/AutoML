import argparse
import ConfigSpace
import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from lccv import LCCV
from IPL import IPL
from surrogate_model import SurrogateModel


def plot_results_schedules(results_dict, save_path, dataset):
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
    plt.title(f"Performance for Different Fixed Schedule Lengths over Iterations on Dataset: {dataset}")

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


def experiment_1(args, anchor_sizes, schedule_length, num_iterations):
    config_space = ConfigSpace.ConfigurationSpace.from_json(args.config_space_file)
    df = pd.read_csv(args.configurations_performance_file)
    surrogate_model = SurrogateModel(config_space)
    surrogate_model.fit(df)

    min_anchor = anchor_sizes[0]
    max_anchor = anchor_sizes[-1]

    # LCCV
    lccv = LCCV(surrogate_model, min_anchor, max_anchor)
    logs_lccv = []
    best_so_far_lccv = None
    predictions_lccv = []

    # IPL
    fixed_schedule = anchor_sizes[:schedule_length]
    ipl = IPL(
        fixed_schedule=fixed_schedule,
        final_anchor=max_anchor,
        best_seen_performance=float('inf'),        
    )   
    logs_ipl = []
    predictions_ipl = []

    for idx in range(num_iterations):
        theta_new = dict(config_space.sample_configuration())

        # LCCV
        result_lccv = lccv.evaluate_model(best_so_far_lccv, theta_new)
        final_result_lccv = result_lccv[-1][1]
        if best_so_far_lccv is None or final_result_lccv < best_so_far_lccv:
            best_so_far_lccv = final_result_lccv
        logs_lccv.append(result_lccv[-1])
        predictions_lccv.append(final_result_lccv)

        # IPL
        result_ipl = []
        for anchor_size in fixed_schedule:
            simulated_performance = surrogate_model.predict(theta_new, anchor_size)
            result_ipl.append((anchor_size, simulated_performance))        
        y_data = [r[1] for r in result_ipl]
        ipl.evaluate_configuration(performances=y_data)
        predictions_ipl.append(ipl.last_seen_prediction)
        
        prediction_true = surrogate_model.predict(theta_new, max_anchor)
        print(f"final_result_lccv: {final_result_lccv}, final_result_ipl: {ipl.last_seen_prediction}, prediction_true: {prediction_true}")
        
 
    errors_lccv = np.abs(predictions_lccv - prediction_true)
    errors_ipl = np.abs(predictions_ipl - prediction_true)

    mean_error_lccv = np.mean(errors_lccv)
    std_error_lccv = np.std(errors_lccv)

    mean_error_ipl = np.mean(errors_ipl)
    std_error_ipl = np.std(errors_ipl)

    print(f"LCCV - Mean Error: {mean_error_lccv:.4f}, Std: {std_error_lccv:.4f}")
    print(f"IPL - Mean Error: {mean_error_ipl:.4f}, Std: {std_error_ipl:.4f}")

    dataset_characteristics = {
        "Dataset": ["Dataset 6", "Dataset 11", "Dataset 1457"],  # Dataset identifiers
        "Instances": [20000, 625, 1500],
        "Features": [17, 5, 10001],
        "Classes": [26, 3, 50],
        "Numeric_Features": [16, 4, 10000],
        "Error_LCCV": [0.28, 0.27, 0.3],
        "Error_IPL": [0.15, 0.12, 0.09],
    }
    df_dataset = pd.DataFrame(data=dataset_characteristics)
    
    # Separate features and targets
    X = df_dataset[['Instances', 'Features', 'Classes', 'Numeric_Features']]
    y_lccv = df_dataset['Error_LCCV']
    y_ipl = df_dataset['Error_IPL']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_lccv, test_size=0.3, random_state=42)

    # Train Random Forest
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Feature importance
    feature_importances = model.feature_importances_
    for name, importance in zip(X.columns, feature_importances):
        print(f"{name}: {importance:.4f}")
    

def experiment_2(args, dataset, schedule_lengths, num_runs, anchor_dict, num_iterations=50, plot_run=True):
    anchor_sizes = anchor_dict[dataset]
    results_dict = {}

    for length in schedule_lengths:
        print('hoi2')
        aggregated_results = []  # To store multiple runs
        for _ in range(num_runs):  # Run each schedule length 10 times
            config_space = ConfigSpace.ConfigurationSpace.from_json(args.config_space_file)
            
            df = pd.read_csv(args.configurations_performance_file)

            # Initialize surrogate model and fit it with the data
            surrogate_model = SurrogateModel(config_space)
            surrogate_model.fit(df)

            # Define the learning curve schedule
            fixed_schedule = anchor_sizes[:length]
            final_max_anchor = anchor_sizes[-1]
            
            # Initialize the IPLModelEvaluator
            evaluator = IPL(
                fixed_schedule=fixed_schedule,
                final_anchor=final_max_anchor,
                best_seen_performance=float('inf'),        
            )

            logs = []  # Track configurations that passed
            discarded_logs = []  # Track configurations that were discarded

            for idx in range(num_iterations):

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

            aggregated_results.append(evaluator.performance_over_iterations)
            
        # Average results per iteration
        averaged_results = [sum(x) / len(x) for x in zip(*aggregated_results)]
        results_dict[length] = averaged_results
        
    # Plot the results
    plot_results_schedules(results_dict, save_path=f"images\schedule_length_plot_{num_runs}_{num_iterations}_{dataset}.png", dataset=dataset)
                

if __name__ == "__main__":
    DATASET = 1457
    SCHEDULE_LENGTHS = [4,5,6,7]
    NUM_RUNS = 15
    NUM_ITERATIONS = 500
    anchor_dict = {6: [16, 32, 64, 128, 256, 512, 1024, 2048, 8192, 16000],
                   11: [16, 32, 64, 128, 256, 512, 1024, 2048, 8192, 16000],
                   1457: [16, 32, 64, 128, 256, 512, 1024, 2048, 8192, 16000]}  
    
    # root = logging.getLogger()
    # root.setLevel(logging.INFO)

    experiment_1(parse_args(DATASET), anchor_sizes=anchor_dict[DATASET], schedule_length=5, num_iterations=10000)
    # experiment_2(parse_args(), DATASET, SCHEDULE_LENGTHS, NUM_RUNS, anchor_dict, num_iterations=NUM_ITERATIONS, plot_run=False)
import argparse
import os
import ConfigSpace as CS
import pandas as pd
from surrogate_model import SurrogateModel
from succesive_halving import SuccesiveHalving


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

    dataset_dict = {"config_performances_dataset-11.csv": 12800,
                    "config_performances_dataset-1457.csv": 6400,
                    "lcdb_configs.csv": 1600,                    
                    "config_performances_dataset-6.csv": 25600,                    
                    }

    for dataset in dataset_dict.keys():
        config_space = CS.ConfigurationSpace.from_json(args.config_space_file)
        df = pd.read_csv(os.path.join(dataset_dir, dataset))

        surrogate_model_external = SurrogateModel(config_space)
        surrogate_model_external.fit(df)

        initial_anchor_size = 25
        max_anchor_size = dataset_dict[dataset]

        start = initial_anchor_size
        target = max_anchor_size
        number_of_halvings = 0

        while start <= target:
            start *= 2
            number_of_halvings += 1

        # To only have 1 config after the last halving
        num_initial_samples = 2**(number_of_halvings-1)
        reduction_factor = 2
        succesive_halving = SuccesiveHalving(surrogate_model_external, initial_anchor_size, max_anchor_size, reduction_factor)

        sample_configs = config_space.sample_configuration(size=num_initial_samples)
        succesive_halving.config_queue = sample_configs
        
        # # succesive_halving.initialize()
        # best_configs = succesive_halving.select_best_configurations()
        # succesive_halving.update_queue(best_configs)

        while succesive_halving.anchor_size <= max_anchor_size: 
            best_configs = succesive_halving.select_best_configurations()
            succesive_halving.update_queue(best_configs)

        if include_plot:
            succesive_halving.plot()
        
        if include_best_performance:
            filtered_configs = {k: v for k, v in succesive_halving.performance_history.items() if len(v) == number_of_halvings-1}
            if filtered_configs:
                best_config = min(filtered_configs, key=lambda k: filtered_configs[k][-1])
                best_performance = filtered_configs[best_config]

                return best_performance  
            
if __name__ == '__main__':
    run_successive_halving(parse_args(), True, False)

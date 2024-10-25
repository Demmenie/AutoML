import argparse
import ConfigSpace as CS
import matplotlib.pyplot as plt
import pandas as pd
from surrogate_model import SurrogateModel
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
    parser.add_argument('--num_iterations', type=int, default=500)

    return parser.parse_args()

def generate_configs(config_space, surrogate_model, with_performance, num_configs=5):
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



def run_smbo(args):
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
    config_space = CS.ConfigurationSpace.from_json(args.config_space_file)
    df = pd.read_csv(args.configurations_performance_file)

    surrogate_model_external = SurrogateModel(config_space)
    surrogate_model_external.fit(df)

    capital_phi = generate_configs(config_space, surrogate_model_external, with_performance=True)
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

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(error_list)), error_list, label='Best Performance Over Time', color='blue')
    plt.xlabel('Iteration')
    plt.ylabel('Best Performance')
    plt.title('Best Performance Over Iterations')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    run_smbo(parse_args())


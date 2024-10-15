import argparse
import os
import ConfigSpace as CS
import matplotlib.pyplot as plt
import pandas as pd
from random_search import RandomSearch
from surrogate_model import SurrogateModel
from succesive_halving import SuccesiveHalving


def df_to_typed_list(df: pd.DataFrame) -> list:
    # Extract the 'score' column name (last column)
    score_column = df.columns[-1]
    
    # Create the list of tuples (dict, float)
    result = [
        (row.drop(score_column).to_dict(), row[score_column]) for _, row in df.iterrows()
    ]
    
    return result

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_space_file', type=str, default='lcdb_config_space_knn.json')
    parser.add_argument('--configurations_performance_file', type=str, default='lcdb_configs.csv')
    # max_anchor_size: connected to the configurations_performance_file. The max value upon which anchors are sampled
    parser.add_argument('--max_anchor_size', type=int, default=1600)
    parser.add_argument('--num_iterations', type=int, default=500)

    return parser.parse_args()

def generate_configs(config_space, surrogate_model, with_performance, num_configs=10):
    configs = []
    # Generate random configurations
    for _ in range(num_configs):
        config = dict(config_space.sample_configuration())  
        if with_performance:     
            performance = surrogate_model.predict(config)               
            configs.append((config, performance))    
        else:
            configs.append((config))  
    return configs


def run(args):
    dataset_dir = "config-performances"
    
    datasets = [f for f in os.listdir(dataset_dir) if f.endswith('.csv')]

    for dataset in datasets:
        config_space = CS.ConfigurationSpace.from_json(args.config_space_file)
        df = pd.read_csv(os.path.join(dataset_dir, dataset))

        surrogate_model_external = SurrogateModel(config_space)
        surrogate_model_external.fit(df)

        # initial_configs = generate_configs(config_space, surrogate_model_external, with_performance=False)
        initial_anchor_size = 100
        max_anchor_size = args.max_anchor_size
        reduction_factor = 2
        succesive_halving = SuccesiveHalving(surrogate_model_external, initial_anchor_size, max_anchor_size, reduction_factor) 

        sample_configs = config_space.sample_configuration(size=100)
        succesive_halving.config_queue = sample_configs
        while succesive_halving.anchor_size <= max_anchor_size:
            best_configs = succesive_halving.select_best_configurations()
            succesive_halving.update_queue(best_configs)    
                
            

    # plt.figure(figsize=(10, 6))
    # plt.plot(range(len(error_list)), error_list, label='Best Performance Over Time', color='blue')
    # plt.xlabel('Iteration')
    # plt.ylabel('Best Performance')
    # plt.title('Best Performance Over Iterations')
    # plt.grid(True)
    # plt.legend()
    # plt.show()


if __name__ == '__main__':
    run(parse_args())
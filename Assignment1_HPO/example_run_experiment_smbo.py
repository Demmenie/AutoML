import argparse
import ConfigSpace as CS
import matplotlib.pyplot as plt
import pandas as pd
from random_search import RandomSearch
from surrogate_model import SurrogateModel
from smbo import SequentialModelBasedOptimization


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
    parser.add_argument('--configurations_performance_file', type=str, default='config-performances\lcdb_configs.csv')
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
    config_space = CS.ConfigurationSpace.from_json(args.config_space_file)
    df = pd.read_csv(args.configurations_performance_file)

    surrogate_model_external = SurrogateModel(config_space)
    surrogate_model_external.fit(df)

    capital_phi = generate_configs(config_space, surrogate_model_external, with_performance=True)
    surrogate_model_internal = SequentialModelBasedOptimization() 
    surrogate_model_internal.initialize(capital_phi)
    error_list = []

    for _ in range(20):
        surrogate_model_internal.fit_model()
        sample_config = config_space.sample_configuration(size=5)
        sample_config = pd.DataFrame(sample_config)
        sample_config['anchor_size'] = args.max_anchor_size
        
        surrogate_model_internal.theta = sample_config
        theta_new = dict(surrogate_model_internal.select_configuration())
        theta_new['anchor_size'] = args.max_anchor_size
        performance = surrogate_model_external.predict(theta_new)
        surrogate_model_internal.update_runs((theta_new, performance))
        error_list.append(surrogate_model_internal.theta_inc_performance)
        print(surrogate_model_internal.theta_inc_performance)

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(error_list)), error_list, label='Best Performance Over Time', color='blue')
    plt.xlabel('Iteration')
    plt.ylabel('Best Performance')
    plt.title('Best Performance Over Iterations')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    run(parse_args())


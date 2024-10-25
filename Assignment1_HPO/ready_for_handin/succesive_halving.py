import heapq
import numpy as np
import matplotlib.pyplot as plt

class SuccesiveHalving(object):
    def __init__(self, surrogate_model, anchor_size, max_anchor_size, reduction_factor):
        self.surrogate_model = surrogate_model
        self.anchor_size = anchor_size
        self.initial_anchor_size = anchor_size
        self.max_anchor_size = max_anchor_size
        self.reduction_factor=reduction_factor
        self.config_queue = []
        self.performance_history = {}

    def initialize(self):
        performance_list = []
        for idx, sample_config in enumerate(self.config_queue):
            sample_config = dict(sample_config)            
            sample_config['anchor_size'] = self.anchor_size
            
            performance = self.surrogate_model.predict(sample_config)
            performance_list.append(performance)
        
            config_key = str(self.config_queue[idx]) 
            if config_key not in self.performance_history:
                self.performance_history[config_key] = [] 
            self.performance_history[config_key].append(performance) 
            
    def select_best_configurations(self):
        performance_list = []

        for idx, sample_config in enumerate(self.config_queue):
            sample_config = dict(sample_config)            
            sample_config['anchor_size'] = self.anchor_size
            
            performance = self.surrogate_model.predict(sample_config)
            performance_list.append(performance)
        
            config_key = str(self.config_queue[idx]) 
            if config_key not in self.performance_history:
                self.performance_history[config_key] = [] 
            self.performance_history[config_key].append(performance)  

        best_indeces = self.find_lowest_n_indices(performance_list, len(performance_list)/self.reduction_factor)
        best_configs = [self.config_queue[i] for i in best_indeces]
        
        return best_configs

    @staticmethod
    def find_lowest_n_indices(performance_list, n):
        n = int(n)

        n_largest_values = heapq.nsmallest(n, enumerate(performance_list), key=lambda x: x[1])    
        indices = [i for i, _ in n_largest_values]

        return indices

    def update_queue(self, best_configs):
        self.config_queue = best_configs
        if self.anchor_size == 0:
            self.anchor_size = 100
        else:
            self.anchor_size = self.anchor_size * 2
        
    def get_performance_history(self):
        return self.performance_history
    
    def plot(self):        
        anchor_sizes = [25,50,100,200,400,800,1600]
        config_labels = {}
        for idx, config_key in enumerate(self.performance_history.keys()):
            config_labels[config_key] = f"Config {idx+1}"

        plt.figure(figsize=(12, 8))

        for config_key, performance_values in self.performance_history.items():
            y_values = np.array(performance_values)
            x_values = anchor_sizes[:len(y_values)]

            y_values = [val if not np.isnan(val) else None for val in y_values]

            plt.plot(x_values, y_values, marker='o', label=config_labels[config_key])

        plt.xlabel('Anchor Size')
        plt.ylabel('Performance')
        plt.title('Performance of Configurations over Anchor Sizes')
        plt.legend(loc='best', fontsize='small', ncol=2)  
        plt.grid(True)
        plt.tight_layout()

        plt.show()
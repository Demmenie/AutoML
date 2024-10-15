import heapq

class SuccesiveHalving(object):
    def __init__(self, surrogate_model, anchor_size, max_anchor_size, reduction_factor):
        self.surrogate_model = surrogate_model
        self.anchor_size = anchor_size
        self.max_anchor_size = max_anchor_size
        self.reduction_factor=reduction_factor
        self.config_queue = []

    def initialize(self):
        return
    
    def select_best_configurations(self):
        performance_list = []

        for sample_config in self.config_queue:
            sample_config = dict(sample_config)            
            sample_config['anchor_size'] = self.anchor_size
            performance = self.surrogate_model.predict(sample_config)
            performance_list.append(performance)

        best_indeces = self.find_highest_n_indices(performance_list, len(performance_list)/self.reduction_factor)
        best_configs = [self.config_queue[i] for i in best_indeces]
        
        return best_configs

    @staticmethod
    def find_highest_n_indices(performance_list, n):
        n = int(n)
        n_largest_values = heapq.nlargest(n, performance_list)
    
        indices = [i for i, value in enumerate(performance_list) if value in n_largest_values]
    
        return indices

    def update_queue(self, best_configs):
        self.queue = best_configs
        self.anchor_size = self.anchor_size*2
        


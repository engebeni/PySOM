import numpy as np

class SOM(object):

    def __init__(self, in_map_height, in_map_width, in_weight_dimensions):
        """
        Constructs a self-organizing map with the specified height and width. Each node's weight vector is
        initialized with uniformly distributed random values between 0 and 1.
        
        in_map_height: The height of the map
        in_map_width: The width of the map
        in_weight_dimensions: The dimension of the weight vectors
        """
        self.shape = (in_map_height, in_map_width, in_weight_dimensions)
        
        #initialization of the map with uniformly distributed random values between 0 and 1
        self.som = np.random.uniform(0, 1, (in_map_height, in_map_width, in_weight_dimensions))
        
        #initialization of training parameters
        self._initial_approx_rate = 0.0
        self._approximation_scale = 0.0
        self._initial_neighborhood_influence_radius = 0.0
        self._neighborhood_radius_scale = 0.0

    def train(self, in_training_data, in_iterations, in_approx_rate, in_approximation_scale, in_neighborhood_influence_radius, in_neighborhood_radius_scale):
        """
        Trains the map with the specified training data sample. The training is performed in iteration cycles, the number of total iterations
        must be specified.
        In each repition the following steps are performed:
        - A random sample s is chosen from the set of training data
        - The best matching unit for s is queried
        - The bmu and its neighborhood (i.e. their weight vectors) are updated
        
        in_training_data: The data which is used to train the map
        in_iterations: The number of iterations for this training
        in_approx_rate: The initial approximation rate, which will decrease with each iteration
        in_approximation_scale: The scale with which the approximation rate will decrease over time
        in_neighborhood_influence_radius: The initial neighbor influence radius, which will decrease with each iteration
        in_neighborhood_radius_scale: The scale with which the neighbor influence radius will decrease over time
        """
        self._initial_approx_rate = in_approx_rate
        self._approximation_scale = in_approximation_scale
        self._initial_neighborhood_influence_radius = in_neighborhood_influence_radius
        self._neighborhood_radius_scale = in_neighborhood_radius_scale
    
        for t in range(in_iterations):
            #choose a random sample from the set of training data
            index_rnd_sample =  np.random.choice(range(len(in_training_data)))
            
            #TODO work with modulo to wrap clusters around edges of the map
            #find the bmu for the sample and update the map
            bmu = self.get_bmu(in_training_data[index_rnd_sample])
            self.update_som(bmu, in_training_data[index_rnd_sample],t)       

    def get_bmu(self, in_sample_vector):
        """
        Returns a tuple with the coordinates of the best matching unit (bmu) for the specified vector.
        The bmu is defined as the node whose corresponding weight vector is closest to the sample vector,
        according to the squared euclidian distance between the two vectors.
        
        in_sample_vector: The weight vector whose bmu shall be found
        """
        som_sum = ((self.som - in_sample_vector) ** 2).sum(axis=2)
        return np.unravel_index(np.argmin(som_sum,axis=None), som_sum.shape)

    def approximation_rate(self, in_current_iteration):
        """
        Returns the approximation rate, that determines how heavily the BMU and its neighbors
        should be pulled toward the respective training sample.
        The approximation rate is a monotonically decreasing function and depends on the
        current iteration and the 
        
        in_current_iteration: The current iteration 
        """
        approx_rate = self._initial_approx_rate * np.exp(-in_current_iteration/self._approximation_scale)
        return approx_rate

    #TODO optimize performance
    def update_som(self, in_bmu, in_sample_vector, in_iteration_cycle):
        """
        Iterates over the whole map, measures the node's distance to the bmu and calls the update_node function with this distance
        
        in_bmu: The bmu for the specified sample vector
        in_sample_vector: The vector which influences the bmu and its neighborhood
        in_iteration_cycle: The current iteration cycle
        """
        for y in range(self.shape[0]):
            for x in range(self.shape[1]):
                dist_to_bmu = np.linalg.norm((np.array(in_bmu) - np.array((y, x))))
                self.update_node((y,x), dist_to_bmu, in_sample_vector, in_iteration_cycle)

    def update_node(self, in_node, in_dist_to_bmu, in_sample_vector, in_iteration_cycle):
        """
        Updates the weight vector of the specified node, depending on its distance to the bmu and the current iteration cycle.
        
        in_node: A tuple containing the coordinates of the specified node
        in_dist_to_bmu: The distance to the bmu
        in_sample_vector: The vector which influences this node
        in_iteration_cycle: The current iteration cycle
        """
        self.som[in_node] = self.som[in_node] + self.distance_penalty(in_dist_to_bmu, in_iteration_cycle)*self.approximation_rate(in_iteration_cycle)*(in_sample_vector - self.som[in_node])

    def distance_penalty(self, in_dist_to_bmu, in_iteration_cycle):
        """
        Computes the distance penalty which affects the node's weight vector value update, depending on the
        distance to the bmu and the current iteration cycle.
        
        in_dist_to_bmu: The distance to the bmu
        in_iteration_cycle: The current iteration cycle
        """
        neighbor_influence_radius = self.neighborhood_influence_radius(in_iteration_cycle)
        dist_penalty = np.exp(-(in_dist_to_bmu ** 2)/(2 * neighbor_influence_radius ** 2))
        return dist_penalty

    def neighborhood_influence_radius(self, in_iteration_cycle):
        """
        Returns the neighborhood influence radius, that determinies which elements in the
        proximity of the BMU are updated.
        The neighborhood influence radius is a monotonically decreasing function and depends on the
        current iteration and the 
        
        in_iteration_cycle: The current iteration cycle
        """
        neighbor_influence_radius = self._initial_neighborhood_influence_radius * np.exp(-in_iteration_cycle/self._neighborhood_radius_scale)
        return neighbor_influence_radius

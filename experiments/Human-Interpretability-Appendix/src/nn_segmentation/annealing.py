# annealing.py
# Functions for performing simulated annealing, temperature generation and neighbor generation
import numpy as np
from statistics import mean

def anneal(start, cost_function, random_neighbour, acceptance_probability, temperature, max_steps=1000, verbose = False):
    '''
    Perform simulated annealing:
    Args:
        start : 
            The starting point of the function, same format as the states
        cost_function : fctn handle :
            The cost function to use. Takes 1 argument (state), returns float or int
        random_neighbour : fctn handle) : 
            Function for generating a random neighbor. Takes 1 argument (state), returns new state proposal
        acceptance_probability : fctn handle : 
            Function to get acceptance probability. Takes 3 arguments (old_cost, new_cost, temperature), returns float in [0,1]
        temperature : fctn handle :
            Function to get current temperature. Takes 1 argument(fraction of steps done), returns float
        max_steps : int (optional) :
            The maximum number of steps the annealing algorithm is allowed to perform
        verbose : bool (optional) :
            Verbosity
    Returns:
        state :
            The final state
        cost :
            The final cost
        statepath :
            The path of states
        costpath :
            The path of costs
    '''
    state = start
    cost = cost_function(state)
    states, costs = [state], [cost]
    for step in range(max_steps):
        fraction = step / float(max_steps)
        T = temperature(fraction)
        new_state = random_neighbour(state)
        new_cost = cost_function(new_state)
        if verbose: print("Step #{:>2}/{:>2} : T = {:>4.3g}, state = {:>4.3g}, cost = {:>4.3g}, new_state = {:>4.3g}, new_cost = {:>4.3g} ...".format(step, max_steps, T, state, cost, new_state, new_cost))
        if acceptance_probability(cost, new_cost, T) > np.random.random():
            state, cost = new_state, new_cost
            states.append(state)
            costs.append(cost)
    return state, cost_function(state), states, costs

def neighbor_flipbit(X):
    '''
    Create a neighbor by flipping a bit

    Args:
        X : np.ndarray :
            Array on which to perform the bit flip. Must be of type float and only contain 1s and 0s
    Returns:
        X_new : np.ndarray :
            New array generated, of type float with a random bit flipped
    '''
    dimension = len(X.shape)
    bit_index = tuple(np.random.randint(0, X.shape[i]) for i in range(dimension))
    X_new = np.copy(X)
    X_new[bit_index] = float(not bool(X_new[bit_index]))
    return X_new

def acceptance_probability_exponential(cost, new_cost, temperature):
    '''
    Return acceptance probability based on exponential method.

    Args:
        cost : float :
            Cost of current solution
        new_cost : float :
            Cost of new solution
        temperature : float :
            Currrent temperature

    Returns:
        p : float :
            Probability of accepting the new solution
    '''
    if new_cost < cost:
        return 1
    else:
        return np.exp(- (new_cost - cost) / temperature)

class Temperature_Generator:
    '''
    Base Class for creating temperature.
    '''
    def __call__(self, fraction):
        raise NotImplementedError('Temperature_Generator is an abstract base class and not intended for use.')

    def get_initial_temperature(self, cost_function, neighbor_generator, starting_point, acceptance_probability, start_iterations):
        increase_cost = []
        decrease_cost = []
        start_cost = cost_function(starting_point)
        for _ in range(start_iterations):
            new_point = neighbor_generator(starting_point)
            delta_cost = cost_function(new_point) - start_cost
            if delta_cost<0:
                decrease_cost.append(delta_cost)
            else:
                increase_cost.append(delta_cost)
        
        m1 = len(decrease_cost)
        m2 = len(increase_cost)
        delta_f = mean(increase_cost)

        return delta_f/np.log(m2/(m2*acceptance_probability-m1*(1-acceptance_probability)))



class Temperature_Generator_linear(Temperature_Generator):
    '''
    Class for creating temperature based on linear cooling schedule. 
    '''
    def __init__(self, mintemp = 0.1, c0=1.0, cost_function=None, neighbor_generator=None, starting_point=None, acceptance_probability=0.8, start_iterations=1000):
        '''
        Class initialisation. For the starting point, you can eiter set c0, or set c0=None and provide cost_function, neighbor_generator, starting_point 
        and acceptance_probability. In that case the starting point is estimated.
        
        Args:
            mintemp : float (optional, default 0.1) :
                Minimum temperature to be returned, to avoid temperatures that are too low for numerical stability
            c0 : float (optional, default 1.0) :
                Starting Temperature
            cost_function : fctn handle (optional, default None) :
                Function for calculating cost
            neighbor_generator : fctn handle (optional, default None) :
                Function for generating a neighboring point
            starting_point : (optional) :
                Starting point of the algorithm
            acceptance_probability : float (optional, default 0.8) :
                Desired acceptance probability
            start_iterations : int (optional, default 1000) :
                Number of points to generate for estimating starting point
        '''
        self._mintemp = mintemp
        if not c0:
            self.c0 = self.get_initial_temperature(cost_function, neighbor_generator, starting_point, acceptance_probability, start_iterations)
        else:
            self.c0 = c0
        super().__init__()

    def __call__(self, fraction):
        max(self._mintemp, self.c0min(1, 1 - fraction))

class Temperature_Generator_geometric(Temperature_Generator):
    '''
    Class for creating temperature based on geometric cooling schedule.
    '''
    def __init__(self, c0=1, alpha=0.95, cost_function=None, neighbor_generator=None, starting_point=None, acceptance_probability=0.8, start_iterations=1000):
        '''
        Class initialisation. For the starting point, you can eiter set c0, or set c0=None and provide cost_function, neighbor_generator, starting_point 
        and acceptance_probability. In that case the starting point is estimated.
        
        Args:
            c0 : float (optional, default 1.0) :
                Starting Temperature
            alpha : float (optional, default 0.95) :
                Decreasing factor for cooling. A factor in [0.8, 0.99] is recommended.
            cost_function : fctn handle (optional, default None) :
                Function for calculating cost
            neighbor_generator : fctn handle (optional, default None) :
                Function for generating a neighboring point
            starting_point : (optional) :
                Starting point of the algorithm
            acceptance_probability : float (optional, default 0.8) :
                Desired acceptance probability
            start_iterations : int (optional, default 1000) :
                Number of points to generate for estimating starting point
        '''
        if not c0:
            self.c = self.get_initial_temperature(cost_function, neighbor_generator, starting_point, acceptance_probability, start_iterations)
        else:
            self.c = c0
        self.alpha = alpha
        super().__init__()

    def __call__(self, fraction):
        current_temp = self.c
        self.c = self.c * self.alpha
        return current_temp
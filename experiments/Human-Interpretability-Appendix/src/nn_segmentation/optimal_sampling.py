# optimal_sampling.py
# Implementations for A&D-Optimal Sampling with different optimization methods

import numpy as np

import annealing

# Optimality methods

class A_Optimality:
    def loss(self, X):
        '''
        Get the A-Optimality Loss

        Args :
            X : np.ndarray :
                The samples aggregated in row form, excluding the '1' for the intercept term
        
        Returns :
            l : float :
                loss score
        '''
        X_ = np.hstack((np.ones((X.shape[0],1)),X))
        if np.any(X_.mean(axis=1)<0.7):
            return np.inf
        try:
            A = np.linalg.inv(X_.T @ X_)
            return np.trace(A)
            return np.trace(A)/A.shape[0]
        except np.linalg.LinAlgError:
            return np.inf
    
    def loss_new_point(self, X, v):
        '''
        Get the A-Optimality Loss for adding a new point. Note that the matrix X_.T @ X_ must be invertible for this loss to apply (X_=[1s, X]). I recommend hot-starting with 
        points that satisfy this condition for example X = I with a row of 0s added at the bottom.

        Args :
            X : np.ndarray :
                The samples aggregated in row form, excluding the '1' for the intercept term
            v : np.ndarray :
                The new point to add
        
        Returns :
            l : float :
                -Maximization objective. This loss can therefore be minimized.
        '''
        X_ = np.hstack((np.ones((X.shape[0],1)),X))
        if np.any(X_.mean(axis=1)<0.7):
            return np.inf
        A = np.linalg.inv(X_.T @ X_)
        v_ = np.hstack((np.array([1]), v))
        return -(v_.T @ A @ A @ v_)/(1+v_.T@A@v_)

class D_Optimality:
    def loss(self, X, tol=1e-10):
        '''
        Get the D-Optimality Loss

        Args :
            X : np.ndarray :
                The samples aggregated in row form, excluding the '1' for the intercept term
            tol : float (optional) :
                The tolerance to say a number is equa to 0
        Returns :
            l : float :
                loss score
        '''
        X_ = np.hstack((np.ones((X.shape[0],1)),X))
        d = np.linalg.det(X_.T @ X_)
        if d>tol:
            return 1/d
            return d**(-X_.shape[1])
        else:
            return np.inf
    
    def loss_new_point(self, X, v):
        '''
        Get the D-Optimality Loss for adding a new point. Note that the matrix X_.T @ X_ must be invertible for this loss to apply (X_=[1s, X]). I recommend hot-starting with 
        points that satisfy this condition for example X = I with a row of 0s added at the bottom.

        Args :
            X : np.ndarray :
                The samples aggregated in row form, excluding the '1' for the intercept term
            v : np.ndarray :
                The new point to add
        
        Returns :
            l : float :
                -Maximization objective. This loss can therefore be minimized.
        '''
        X_ = np.hstack((np.ones((X.shape[0],1)),X))
        A = np.linalg.inv(X_.T @ X_)
        v_ = np.hstack((np.array([1]), v))
        return -(v_.T @ A @ v_)

# Optimization methods

def find_X_random_candidate(optimality, n_samples, X, p_binomial=0.5, n_candidates=100):
    '''
    Build X with a greedy approach. At each iteration, consider n_samples random points to add to X, add the best one.

    Args:
        optimality : class instance :
            Optimality method. Must implement loss_new_point(X,v), the loss when adding v to X
        n_samples : int :
            Number of samples in the final matrix X
        X : np.ndarray :
            Samples for starting. Note that many Optimality methods require that X_.T @ X_ must be invertible (X_=[1s, X])
        p_binomial : float in [0,1] (optional):
            Probability for each entry of candidates to be equal to 1
        n_candidates : int (optional) :
            The number of candidates to consider at each iteration
    
    Returns:
        X : np.ndarray :
            The final matrix of samples
    '''
    for i in range(X.shape[0], n_samples):
        # find the best vector
        v_best = None
        best_loss = np.inf
        for _ in range(n_candidates):
            v = np.random.binomial(1, p_binomial, size=X.shape[1])
            curr_loss = optimality.loss_new_point(X, v)
            if curr_loss<best_loss:
                best_loss = curr_loss
                v_best = v
        # add the vector
        X = np.vstack((X, np.array(v_best)))
    return X

def find_X_enumerate_candidates(optimality, n_samples, possibilities, X):
    '''
    Build X with a greedy approach. At each iteration, consider all possible samples to add to X, add the best one.

    Args:
        optimality : class instance :
            Optimality method. Must implement loss_new_point(X,v), the loss when adding v to X
        possibilities : np.ndarray:
            possibilities to add to X
        X : np.ndarray :
            Samples for starting. Note that many Optimality methods require that X_.T @ X_ must be invertible (X_=[1s, X])
    
    Returns:
        X : np.ndarray :
            The final matrix of samples
    '''
    for i in range(X.shape[0], n_samples):
        # find the best vector
        v_best = None
        best_loss = np.inf
        for v in possibilities:
            curr_loss = optimality.loss_new_point(X, v)
            if curr_loss<best_loss:
                best_loss = curr_loss
                v_best = v
        # add the vector
        X = np.vstack((X, np.array(v_best)))
    return X

def find_X_annealing(optimality, n_samples, random_neighbour, acceptance_probability, temperature, X, p_binomial=0.5, max_steps=1000, T0=1):
    for i in range(X.shape[0], n_samples):
        # generate a random starting point for the annealing algorithm
        v = np.random.binomial(1, p_binomial, size=X.shape[1])
        # wrapper of the optimality function for the annealing algorithm
        def cost_wrapper(v_i):
            return optimality.loss_new_point(X, v_i)
        # perform the annealing
        if T0:
            T = temperature(c0=T0)
        else:
            T = temperature(c0=None, cost_function=cost_wrapper, neighbor_generator=random_neighbour, starting_point=v)
        v, _, _, _ = annealing.anneal(v, cost_wrapper, random_neighbour, acceptance_probability, T, max_steps=max_steps)
        X = np.vstack((X, np.array(v)))
    return X

def find_X_annealing_direct(optimality, n_samples, random_neighbour, acceptance_probability, temperature, X, p_binomial=0.5, max_steps=1000, T0=1):
    X = np.vstack((X, np.random.binomial(1, p_binomial, size=(n_samples - X.shape[0], X.shape[1]))))
    if T0:
        T = temperature(c0=T0)
    else:
        T = temperature(c0=None, cost_function=optimality.loss, neighbor_generator=random_neighbour, starting_point=X)
    X, _, _, _ = annealing.anneal(X, optimality.loss, random_neighbour, acceptance_probability, T, max_steps=max_steps)
    return X

def initial_X_binary(dimension):
    X = np.ones((dimension, dimension)) - np.eye(dimension)
    X = np.vstack((X, np.ones((1,dimension))))
    return X
        
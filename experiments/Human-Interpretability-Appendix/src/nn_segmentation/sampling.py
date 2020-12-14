# sampling.py
# Implements methods for sampling

import numpy as np
import pandas as pd

import optimal_sampling
import annealing


def sampling_random_topic(chunks, n_samples, p_binomial=0.7, p_topic=0.5):
    samples = []
    for _ in range(n_samples):
        if np.random.random()<p_topic:
            # sample from a topic
            topic = np.random.choice(chunks)
            s = np.array([float(c==topic) for c in chunks])
        else:
            # sample randomly
            s = np.random.binomial(1, p_binomial, size=len(chunks))
        samples.append(s)
    return np.vstack(tuple(samples))

def sampling_random(chunks, n_samples, p_binomial=0.7):
    return sampling_random_topic(chunks, n_samples, p_binomial=p_binomial, p_topic=0)

def sample_D_optimal(chunks, n_samples, max_steps=100000):
    dimension = len(chunks)
    X = optimal_sampling.initial_X_binary(dimension)
    X = optimal_sampling.find_X_annealing_direct(optimal_sampling.D_Optimality(),
                                n_samples,
                                annealing.neighbor_flipbit, 
                                annealing.acceptance_probability_exponential,
                                annealing.Temperature_Generator_geometric,
                                X,
                                max_steps=max_steps)
    return X

def sample_A_optimal(chunks, n_samples, max_steps=100000):
    dimension = len(chunks)
    X = optimal_sampling.initial_X_binary(dimension)
    X = optimal_sampling.find_X_annealing_direct(optimal_sampling.A_Optimality(),
                                n_samples,
                                annealing.neighbor_flipbit, 
                                annealing.acceptance_probability_exponential,
                                annealing.Temperature_Generator_geometric,
                                X,
                                max_steps=max_steps)
    return X
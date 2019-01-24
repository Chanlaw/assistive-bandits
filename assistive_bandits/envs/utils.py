from __future__ import division

import math
import numpy as np
from scipy.stats import bernoulli, uniform, beta

import gym
from gym.spaces import prng

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class beta_bernoulli_iterator:
    """Iterator for drawing parameters for a bernoulli distribution from a beta distribution"""
    def __init__(self, a=2, b=2):
        self.a = a
        self.b = b
    
    def __iter__(self):
        return self
    
    def __next__(self):
        return {'p': beta.rvs(self.a,self.b)}

class uniform_bernoulli_iterator:
    """Iterator for drawing parameters for a bernoulli distribution from a uniform distribution"""
    def __init__(self):
        self.min=0
        self.max=1
    
    def __iter__(self):
        return self
    
    def __next__(self):
        return {'p': uniform.rvs()}

class dummy_iterator:
    def __init__(self):
        self.count = 0
        self.theta = np.random.randint(0,2)
        
    def __iter__(self):
        return self
    
    def __next__(self):
        p = 1.0 if self.theta == self.count else 0.0
        self.count += 1
        if self.count >= 2:
            self.count = 0
            self.theta = np.random.randint(0,2)
        return {'p': p}
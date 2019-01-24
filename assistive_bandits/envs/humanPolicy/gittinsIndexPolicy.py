from __future__ import print_function, division

import time
import numpy as np
import tensorflow as tf
import rllab.misc.logger as logger

from sandbox.rocky.tf.spaces.discrete import Discrete #supercedes the gym spaces
from assistive_bandits.envs.bandit import BanditEnv
from assistive_bandits.envs.utils import softmax
from assistive_bandits.envs.humanPolicy.humanPolicy import HumanPolicy


def initial_approximation(pulls, discount, grid_n):
    """Approximate the initial values for the value function to begin backward induction.

    Pulls specifies the total number of bandit arm pulls and observations from which backward
    induction is used to compute the index values for any distribution of discrete binary
    observations. Success denoted by a, and failure denoted by b.

    Assumptions 1 <= a,b <= pulls - 1, so we assume at least one observation of success and failure.
    """

    values = np.zeros([pulls - 1, pulls - 1, grid_n])  # Store V(a=k, b=n-k, r) in values[k,n-1,:] as k varies
    gittins = np.zeros([pulls - 1, pulls - 1])  # Store Gittins(a=k, b=n-k) in gittins[k,n-1] as k varies

    a_grid = np.arange(1, pulls)
    r_grid = np.linspace(0, 1, grid_n)

    initial_gittins = a_grid / float(pulls)  # Initial Gittins Approximation to start Backward Induction
    gittins[0:pulls, pulls - 2] = initial_gittins  # Record initial Gittins approximation

    for idx_a, a in enumerate(a_grid):
        values[idx_a, pulls - 2, :] = (1.0 / (1 - discount)) * \
                                      np.maximum(r_grid, a / float(pulls))  # Record initial Value approximation

    return gittins, values


def recursion_step(value_n, r_grid, discount):
    """One-step backward induction computing the value function and the Gittins Index.

    Based on the description in Gittins etal 2011 and Powell and Ryzhov 2012.
    """

    n = value_n.shape[0]
    r_len = r_grid.shape[0]
    value_n_minus_1 = np.zeros([n - 1, r_len])  # Value function length reduced by 1
    gittins_n_minus_1 = np.zeros(n - 1)  # Value function length reduced by 1
    for k in range(0, n - 1):
        a = k + 1  # a in range [1,n-1]
        b = n - k - 1  # b in range [1,n-1]
        value_n_minus_1[k, :] = np.maximum((r_grid / float(1 - discount)),
                                           (a / float(n)) * (1 + discount * value_n[k + 1, :]) +
                                           (b / float(n)) * discount * value_n[k, :]
                                           )
        try:
            # Find first index where Value = (Value of Safe Arm)
            idx_git = np.argwhere((r_grid / float(1 - discount)) == value_n_minus_1[k, :]).flatten()
            gittins_n_minus_1[k] = 0.5 * (r_grid[idx_git[0]] + r_grid[idx_git[0] - 1])  # Take average
        except:
            print("Error in finding Gittins index")

    return gittins_n_minus_1, value_n_minus_1


def recursion_loop(pulls, discount, grid_n):
    """This produces the value functions and Gittins indices by backward induction"""

    r_grid = np.linspace(0, 1, grid_n)
    gittins, values = initial_approximation(pulls, discount, grid_n)
    n = pulls - 2  # Note that the 2 comes from (1) the initial approximation and (2) python indexing
    while n >= 1:
        g, v = recursion_step(values[:n + 1, n, :], r_grid, discount)
        values[:n, n - 1] = v
        gittins[:n, n - 1] = g
        n -= 1
    return gittins, values


def reformat_gittins(g, v=None):
    """Reformat output.

    We reformat the result to get the results in a similar form
    as in (Gittins etal 2011, Powell and Ryzhov 2012), except that we store:
    Success count denoted by a in rows
    Failure count denoted by b in columns
    """
    n = g.shape[0]
    g_reformat = np.zeros(g.shape)

    for row in range(0, n):
        g_reformat[row, :n - row] = g[row, row:]
    return g_reformat


def gittins_index(n=100, grid=20000, discount=0.9):
    """Compute Gittins indices and value functions.

    To get the results to match up with Gittins etal. (2011, p.265) 
    or Powell and Ryzhov (2012, p.144-5) we need a fairly fine grid: approx 10000 grid points.
    """
    g, v = recursion_loop(n, discount, grid)
    g_reformat = reformat_gittins(g) 
    return g_reformat

class GIBetaBernoulliBanditPolicy(HumanPolicy):
    """ The greedy policy with respect to the Gittins Index. """
    def __init__(self, env, n_pulls=100, discount=0.9):
        """
        Initialize this policy by computing and storing the Gittins Index, assuming the environment
        is a Beta-Bernoulli bandit. 
        
        Args:
            env: the BanditEnv for the polciy to act in.
            n_pulls: the number of pulls to explicitly compute the Gittins index for
        """
        assert(isinstance(env, BanditEnv))
        self.horizon = env.horizon
        # logger.log("Computing Gittins Index...")
        start_time = time.time()
        n_pulls = max(int(env.horizon*1.5), n_pulls)
        self.Gittins = gittins_index(n_pulls, discount=discount)
        # logger.log("Elapsed time in Gittins Index Calculation: {}".format( time.time() - start_time))
        super(GIBetaBernoulliBanditPolicy, self).__init__(env)
        
    def reset(self):
        self.successes = np.ones(self.env.nA, dtype=np.int32)
        self.failures = np.ones(self.env.nA, dtype=np.int32)
        self.arm_indices = np.array([self.Gittins[1, 1]])
    
    def get_action(self, obs):
        self.arm_indices = np.array([self.Gittins[self.successes[i], self.failures[i]] \
                                     for i in range(self.env.nA)])
        return self.np_random.choice(np.where(np.isclose(self.arm_indices, np.max(self.arm_indices)))[0])
    
    def learn(self, old_obs, act, rew, new_obs, done):
        if rew > 0.5: 
            self.successes[act] += 1
        else:
            self.failures[act] += 1
            
    # Functions for MCMC
    def get_initial_state(self, **kwargs):
        """ Returns a dict of the human's initial internal state """
        return {'successes': np.ones(self.env.nA, dtype=np.int32),
               'failures': np.ones(self.env.nA, dtype=np.int32)}

    def get_action_from_state(self, state, obs):
        raise NotImplementedError

    def update_state(self, state, old_obs, act, rew, new_obs):
        if rew > 0.5: 
            state['successes'][act] += 1
        else:
            state['failures'][act] += 1
        return state

    def log_likelihood(self, state, obs, act):
        """ Computes the log likelihood of taking the action given the state. """
        indices = np.array([self.Gittins[state['successes'][i], state['failures'][i]] for i in range(self.env.n_arms)])
        greedy_arms = np.where(np.isclose(indices,indices.max()))[0]
        return np.log(1/len(greedy_arms)) if act in greedy_arms else -1e8
    
    # Functions for RTDP
    def act_probs_from_counts(self, counts, **kwargs):
        indices = np.array([self.Gittins[counts[2*i], counts[2*i+1]] for i in range(self.env.n_arms)])
        act_probs = np.zeros(self.env.n_arms, dtype=np.float32)
        greedy_arms = np.where(np.isclose(indices,indices.max()))[0]
        act_probs[greedy_arms] += 1/len(greedy_arms)
        return act_probs
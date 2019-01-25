from __future__ import print_function, division

import time
import numpy as np
import tensorflow as tf
import rllab.misc.logger as logger

from sandbox.rocky.tf.spaces.discrete import Discrete #supercedes the gym spaces
from assistive_bandits.envs.bandit import BanditEnv
from assistive_bandits.envs.utils import softmax
from assistive_bandits.envs.humanPolicy.humanPolicy import HumanPolicy

class EpsOptimalBanditPolicy(HumanPolicy):
    """
    Epsilon-greedy policy for Bandit problems, but with access to the true arm means. Choose the 
    highest mean arm with probability 1-epsilon and a random arm with probabiltiy epsilon.
    """

    def __init__(self, env, epsilon = 0.1):
        assert (isinstance(env.observation_space, Discrete) \
                or isinstance(env.observation_space, spaces.Discrete))
        assert (isinstance(env.action_space, Discrete) \
                or isinstance(env.action_space, spaces.Discrete))
        
        self.epsilon = epsilon
        
        super(EpsOptimalBanditPolicy, self).__init__(env)
        
    def reset(self):
        #reset actual mean table
        self.means = np.array([arm.mean() for arm in self.env.arms])
        
    def get_action(self, obs):
        act = self.env.action_space.sample() if self.np_random.random_sample() < self.epsilon \
                            else self.np_random.choice(np.where(np.isclose(self.means,
                                                                  self.means.max()))[0])
        return act
    
    def learn(self, old_obs, act, rew, new_obs, done):
        pass
        
    def get_state(self):
        return {'means': self.means}
    
    def get_initial_state(self, arm_means=None, **kwargs):
        if arm_means is None:
            arm_means = np.array([arm.mean() for arm in self.env.arms])
        return {'means': arm_means}

    def get_action_from_state(self, state, obs):
        act = self.np_random.choice(range(self.env.nA))if self.np_random.random_sample() < self.epsilon \
                            else self.np_random.choice(np.where(np.isclose(state[means],
                                                                  state[means].max()))[0])
        return act

    def update_state(self, state, old_obs, act, rew, new_obs):
        return state

    def likelihood(self, state, obs, act):
        """ Computes the likelihood of taking the action given the state. """
        greedy_arms = np.where(np.isclose(state['means'], state['means'].max()))[0]
        return (1-self.epsilon)/len(greedy_arms)+ self.epsilon/self.env.n_arms \
                    if act in greedy_arms else  self.epsilon/self.env.n_arms
    
    def log_likelihood(self, state, obs, act):
        return np.log(self.likelihood(state, obs, act))
    
    def act_probs_from_counts(self, counts, **kwargs):
        means = np.array([arm.mean() for arm in self.env.arms])
        act_probs = np.full(self.env.n_arms, self.epsilon/self.env.n_arms)
        greedy_arms = np.where(np.isclose(means,means.max()))[0]
        act_probs[greedy_arms] += (1-self.epsilon)/len(greedy_arms)
        return act_probs

class BoltzmannOptimalBanditPolicy(HumanPolicy):
    """
    Boltzmann-rational policy for Bandit problems, with access to the true arm means. Pulls arms
    proportional to the exponent of their mean. 
    """
    def __init__(self, env, softmax_temp=0.2):
        assert (isinstance(env.observation_space, Discrete) \
                or isinstance(env.observation_space, spaces.Discrete))
        assert (isinstance(env.action_space, Discrete) \
                or isinstance(env.action_space, spaces.Discrete))
        
        self.softmax_temp = softmax_temp
        
        super(BoltzmannOptimalBanditPolicy, self).__init__(env)
        
    def reset(self):
        #reset empirical mean table
        self.means = np.array([arm.mean() for arm in self.env.arms])
        
    def get_action(self, obs):
        act_dist = softmax(self.means/self.softmax_temp)
        act = np.random.choice(range(self.env.nA), p=act_dist)
        return act
    
    def learn(self, old_obs, act, rew, new_obs, done):
        pass
        
    def get_state(self):
        return {'means': self.means, 'temp': self.softmax_temp}
    
    #For MCMC-based methods
    def get_initial_state(self, arm_means=None, **kwargs):
        if arm_means is None:
            arm_means = np.array([arm.mean() for arm in self.env.arms])
        return {'means': arm_means, 'temp':self.softmax_temp}

    def get_action_from_state(self, state, obs):
        act_dist = softmax(state['means']/state['temp'])
        act = np.random.choice(range(self.env.nA), p=act_dist)
        return act

    def update_state(self, state, old_obs, act, rew, new_obs):
        return state

    def likelihood(self, state, obs, act):
        """ Computes the likelihood of taking the action given the state. """
        act_dist = softmax(state['means']/state['temp'])
        return act_dist[act]
    
    def log_likelihood(self, state, obs, act):
        return np.log(self.likelihood(state, obs, act))
    
    def act_probs_from_counts(self, counts, **kwargs):
        means = np.array([arm.mean() for arm in self.env.arms])
        act_dist = softmax(means/self.softmax_temp)
        return act_dist


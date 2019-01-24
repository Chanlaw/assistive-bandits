from __future__ import print_function, division

import numpy as np
import tensorflow as tf

import gym
from gym.utils import seeding
from gym import spaces

from sandbox.rocky.tf.spaces.discrete import Discrete #supercedes the gym spaces
from assistive_bandits.envs.utils import softmax
from assistive_bandits.envs.humanPolicy.humanPolicy import HumanPolicy

class KGBetaBernoulliBanditPolicy(HumanPolicy):
    """ 
    Implements a knowledge gradient-based policy for a Beta-Bernoulli Bandit.
    
    Decision rule is:
        argmax_k (theta^t_k + (T - t -1) v^KG,t_k)
        where v^KG is the knowledge gradient. 
    """
    def __init__(self, env):
        assert (isinstance(env.observation_space, Discrete) \
                or isinstance(env.observation_space, spaces.Discrete))
        assert (isinstance(env.action_space, Discrete) \
                or isinstance(env.action_space, spaces.Discrete))
        self.T = env.horizon
        
        super(KGBetaBernoulliBanditPolicy, self).__init__(env)
    
    def reset(self):
        self.t = 0
        self.successes = np.ones(self.env.nA)
        self.failures = np.ones(self.env.nA)
        self.means = self.successes/(self.successes + self.failures)
        self.knowledge_gradient = np.zeros(self.env.nA)
        for i in range(self.env.nA):
            new_means_1 = self.means.copy()
            new_means_2 = self.means.copy()
            new_means_1[i] = (self.successes[i] + 1)/(self.successes[i] + self.failures[i] + 1)
            new_means_2[i] = (self.successes[i])/(self.successes[i] + self.failures[i] + 1)
            expected_new_max_mean = self.means[i] * np.max(new_means_1) + (1-self.means[i])*np.max(new_means_2)
            self.knowledge_gradient[i] = expected_new_max_mean
        self.knowledge_gradient -= np.max(self.means)
    
    def get_action(self, obs):
        self.arm_vals = self.means + (self.T - self.t + 1) * self.knowledge_gradient
        return self.np_random.choice(np.where(np.isclose(self.arm_vals, np.max(self.arm_vals)))[0])
    
    def learn(self, old_obs, act, rew, new_obs, done):
        self.t += 1
        if rew > 0.5: 
            self.successes[act] += 1
        else:
            self.failures[act] += 1
        self.means[act] = self.successes[act]/(self.successes[act] + self.failures[act])
        #update knowledge gradient
        for i in range(self.env.nA):
            new_means_1 = self.means.copy()
            new_means_2 = self.means.copy()
            new_means_1[i] = (self.successes[i] + 1)/(self.successes[i] + self.failures[i] + 1)
            new_means_2[i] = (self.successes[i])/(self.successes[i] + self.failures[i] + 1)
            expected_new_max_mean = self.means[i] * np.max(new_means_1) + (1-self.means[i])*np.max(new_means_2)
            self.knowledge_gradient[i] = expected_new_max_mean
        self.knowledge_gradient -= np.max(self.means)

    #Functions for MCMC
    def get_initial_state(self, **kwargs):
        t = 0
        successes = np.ones(self.env.nA)
        failures = np.ones(self.env.nA)
        means = successes/(successes+failures)
        knowledge_gradient = np.zeros(self.env.nA)
        for i in range(self.env.nA):
            new_means_1 = self.means.copy()
            new_means_2 = self.means.copy()
            new_means_1[i] = (self.successes[i] + 1)/(self.successes[i] + self.failures[i] + 1)
            new_means_2[i] = (self.successes[i])/(self.successes[i] + self.failures[i] + 1)
            expected_new_max_mean = self.means[i] * np.max(new_means_1) + (1-self.means[i])*np.max(new_means_2)
            knowledge_gradient[i] = expected_new_max_mean
        knowledge_gradient -= np.max(means)
        return {'t': t, 'means': means, 'successes': successes, 'failures': failures, 'kg': knowledge_gradient}

    def get_action_from_state(self, state, obs):
        arm_vals = state['means'] + (self.T - self.t + 1) * state['kg']
        act = self.np_random.choice(np.where(np.isclose(arm_vals, arm_vals.max()))[0])
        return act

    def update_state(self, state, old_obs, act, rew, new_obs):
        state['t'] += 1
        if rew > 0.5: 
            state['successes'][act] += 1
        else:
            state['failures'][act] += 1
        state['means'][act] = state['successes'][act]/(state['successes'][act] + state['failures'][act])
        #update knowledge gradient
        for i in range(self.env.nA):
            new_means_1 = state['means'].copy()
            new_means_2 = state['means'].copy()
            new_means_1[i] = (state['successes'][i] + 1)/(state['successes'][i] + state['failures'][i] + 1)
            new_means_2[i] = (state['successes'][i])/(state['successes'][i] + state['failures'][i] + 1)
            expected_new_max_mean = state['means'][i] * np.max(new_means_1) + (1-state['means'][i])*np.max(new_means_2)
            state['kg'][i] = expected_new_max_mean
        state['kg'] -= np.max(state['means'])
        return state
    
    def likelihood(self, state, obs, act):
        """ Computes the likelihood of taking the action given the state. """
        arm_vals = state['means'] + (self.T - self.t + 1) * state['kg']
        greedy_arms = np.where(np.isclose(arm_vals, arm_vals.max()))[0]
        return 1/len(greedy_arms) if act in greedy_arms else 1e-10
    
    def log_likelihood(self, state, obs, act):
        return np.log(self.likelihood(state, obs, act))

    #RTDP
    def act_probs_from_counts(self, counts, *args, **kwargs):
        t = sum(counts) - 2*self.env.nA
        successes = np.array([counts[2*i] for i in range(self.env.nA)])
        failures = np.array([counts[2*i+1] for i in range(self.env.nA)])
        means = successes/(successes + failures)
        
        #calculate knowledge gradient
        knowledge_gradient = np.zeros(self.env.nA)
        for i in range(self.env.nA):
            new_means_1 = means.copy()
            new_means_2 = means.copy()
            new_means_1[i] = (successes[i] + 1)/(successes[i] + failures[i] + 1)
            new_means_2[i] = (successes[i])/(successes[i] + failures[i] + 1)
            expected_new_max_mean = means[i] * np.max(new_means_1) + (1-means[i])*np.max(new_means_2)
            knowledge_gradient[i] = expected_new_max_mean
        knowledge_gradient -= np.max(means)
        arm_vals = means + (self.T - t + 1) * knowledge_gradient
        
        act_probs = np.zeros(self.env.nA, dtype=np.float32)
        greedy_arms = np.where(np.isclose(arm_vals,arm_vals.max()))[0]
        act_probs[greedy_arms] += 1/len(greedy_arms)
        return act_probs
        
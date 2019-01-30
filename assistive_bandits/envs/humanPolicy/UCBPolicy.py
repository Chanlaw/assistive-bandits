from __future__ import print_function, division

import numpy as np
import tensorflow as tf
from scipy.stats import beta

import gym
from gym.utils import seeding
from gym import spaces

from sandbox.rocky.tf.spaces.discrete import Discrete #supercedes the gym spaces
from assistive_bandits.envs.utils import softmax
from assistive_bandits.envs.humanPolicy.humanPolicy import HumanPolicy

class UCBBanditPolicy(HumanPolicy):
	"""
	Basic UCB Algorithm (Lai and Robbins 1985). 

	First pick each arm once, then at time t pick the arm with the highest value of 
			mean + p*sqrt(2(log(t)+2 log log(t))/counts),
	where counts is the number of times each arm has been pulled. 

	No assumption about the distribution of arm rewards are made. 
	"""

	def __init__(self, env, p=1):
        assert (isinstance(env.observation_space, Discrete) \
                or isinstance(env.observation_space, spaces.Discrete))
        assert (isinstance(env.action_space, Discrete) \
                or isinstance(env.action_space, spaces.Discrete))
        self.p = p
        self.horizon = env.horizon
        self.log_horizon = np.log(self.horizon)
        
        super(BayesUCBBetaBernoulliBanditPolicy, self).__init__(env)
    
    def reset(self):
        self.t = 0
        self.counts = np.zeros(self.env.n_arms, dtype=np.float32)
        self.means = np.zeros(self.env.n_arms, dtype=np.float32)

        self.ucb_bonus = np.sqrt(2*(np.log(self.t+1)+2*np.log(np.log(self.t+1)))/(self.counts+1e-10))

        
    def get_action(self, obs):
        self.arm_vals = self.means + self.p*self.ucb_bonus
        act = self.np_random.choice(np.where(np.isclose(self.arm_vals, self.arm_vals.max()))[0])
        return act
    
    def learn(self, old_obs, act, rew, new_obs, done):
        self.t += 1
        self.counts[act] += 1
        self.means[act] = (self.counts[act]-1)*self.means[act]/self.counts[act]+rew/self.counts[act]

        #update UCB bonus
        self.ucb_bonus = np.sqrt(2*(np.log(self.t+1)+2*np.log(np.log(self.t+1)))/(self.counts+1e-10))

class BayesUCBBetaBernoulliBanditPolicy(HumanPolicy):
	"""
	Bayes UCB Bandit algorithm (Kaufmann, Cappe, and Garivier 2012) with Beta-Bernoulli prior. 

	Pick the arm with the highest (1-1/t)-quantile in the posterior. 
	"""
	def __init__(self, env, c=0, prior_a=1, prior_b=1):
        assert (isinstance(env.observation_space, Discrete) \
                or isinstance(env.observation_space, spaces.Discrete))
        assert (isinstance(env.action_space, Discrete) \
                or isinstance(env.action_space, spaces.Discrete))
        self.c=c
        self.prior_a = prior_a
        self.prior_b = prior_b
        self.horizon = env.horizon
        self.log_horizon = np.log(self.horizon)
        
        super(BayesUCBBetaBernoulliBanditPolicy, self).__init__(env)
    
    def reset(self):
        self.t = 0
        self.successes = np.full(self.env.nA, self.prior_a, dtype=np.int32)
        self.failures = np.full(self.env.nA, self.prior_b, dtype=np.int32)
        self.means = self.successes/(self.successes + self.failures)
        #UCB bonus is just the value of the 1-1/(t log(T)^c)-quantile 
        self.ucb_bonus = beta.ppf(1-1/((self.t+1)*self.log_horizon**self.c), self.successes, self.failures) 

    def get_action(self, obs):
        self.arm_vals = self.ucb_bonus
        act = self.np_random.choice(np.where(np.isclose(self.arm_vals, self.arm_vals.max()))[0])
        return act
    
    def learn(self, old_obs, act, rew, new_obs, done):
        self.t += 1
        if rew > 0.5: 
            self.successes[act] += 1
        else:
            self.failures[act] += 1
        self.means[act] = self.successes[act]/(self.successes[act] + self.failures[act])

        #update UCB bonus
        self.ucb_bonus = beta.ppf(1-1/((self.t+1)*self.log_horizon**self.c), self.successes, self.failures) 


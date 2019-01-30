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


class DeterministicUCLBetaBernoulliBanditPolicy(HumanPolicy):
    """
    Upper Credible Limit (Reverdy et al 2014) algorithm for bandit problems with no action selection noise. 
    
    Basically just Bayes-UCB.

    Uses Beta Bernoulli Prior. 
    """
    def __init__(self, env, K=4, prior_a=1, prior_b=1):
        assert (isinstance(env.observation_space, Discrete) \
                or isinstance(env.observation_space, spaces.Discrete))
        assert (isinstance(env.action_space, Discrete) \
                or isinstance(env.action_space, spaces.Discrete))
        self.K = K
        self.prior_a = prior_a
        self.prior_b = prior_b
        
        super(DeterministicUCLBetaBernoulliBanditPolicy, self).__init__(env)
    
    def reset(self):
        self.t = 0
        self.successes = np.full(self.env.nA, self.prior_a, dtype=np.int32)
        self.failures = np.full(self.env.nA, self.prior_b, dtype=np.int32)
        self.means = self.successes/(self.successes + self.failures)
        self.ucl_bonus = beta.ppf(1-1/(self.K*(self.t+1)), self.successes, self.failures)
        
    def get_action(self, obs):
        self.arm_vals = self.means + self.ucl_bonus
        act = self.np_random.choice(np.where(np.isclose(self.arm_vals, self.arm_vals.max()))[0])
        return act
    
    def learn(self, old_obs, act, rew, new_obs, done):
        self.t += 1
        if rew > 0.5: 
            self.successes[act] += 1
        else:
            self.failures[act] += 1
        self.means[act] = self.successes[act]/(self.successes[act] + self.failures[act])

        #update UCL bonus
        self.ucl_bonus = beta.ppf(1-1/(self.K*(self.t+1)), self.successes, self.failures)

    def get_state(self):
        state = {}
        state['t'] = self.t
        state['K'] = self.K
        state['successes'] = self.successes
        state['failures'] = self.failures
        state['means'] = self.means
        state['ucl_bonus'] = self.ucl_bonus
        return state

    #For MCMC-based methods
    def get_initial_state(self, **kwargs):
        state = {}
        state['t'] = 0
        state['K'] = self.K
        state['successes'] = np.full(self.env.nA, self.prior_a, dtype=np.int32)
        state['failures'] = np.full(self.env.nA, self.prior_b, dtype=np.int32)
        state['means'] = state['successes']/(state['successes'] + state['failures'])
        state['ucl_bonus'] = beta.ppf(1-1/(state['K']*(state['t']+1)), state['successes'], state['failures'])
        return state

    def get_action_from_state(self, state, obs):
        arm_vals = state['means'] + state['ucl_bonus']
        act = self.np_random.choice(np.where(np.isclose(arm_vals, arm_vals.max()))[0])
        return act

    def update_state(self, state, old_obs, act, rew, new_obs):
        state['t'] += 1
        if rew > 0.5:
            state['successes'][act] += 1
        else:
            state['failures'][act] += 1
        state['means'][act] = state['successes'][act]/(state['successes'][act] + state['failures'][act]) 

        #update UCL bonus 
        state['ucl_bonus'] = beta.ppf(1-1/(state['K']*(state['t']+1)), state['successes'], state['failures'])
        return state
    
    def likelihood(self, state, obs, act):
        arm_vals = state['means'] + state['ucl_bonus']
        greedy_arms = np.where(np.isclose(arm_vals, arm_vals.max()))[0]
        return 1/len(greedy_arms) if act in greedy_arms else 1e-8

    def log_likelihood(self, state, obs, act):
        return np.log(self.likelihood(state, obs, act))


class UCLBetaBernoulliBanditPolicy(HumanPolicy):
    """
    Upper Credible Limit (Reverdy et al 2014) algorithm for bandit problems with softmax action selection noise. 
    
    Basically just Bayes-UCB, but noisy.
    
    Uses Beta Bernoulli Prior. 
    """
    PPF_cache = {}

    def __init__(self, env, K=4, prior_a=1, prior_b=1, softmax_temp=0.25):
        assert (isinstance(env.observation_space, Discrete) \
                or isinstance(env.observation_space, spaces.Discrete))
        assert (isinstance(env.action_space, Discrete) \
                or isinstance(env.action_space, spaces.Discrete))
        
        self.softmax_temp = softmax_temp
        self.K = K
        self.prior_a = prior_a
        self.prior_b = prior_b
        
        super(UCLBetaBernoulliBanditPolicy, self).__init__(env)
    
    def reset(self):
        self.t = 0
        self.successes = np.full(self.env.nA, self.prior_a, dtype=np.int32)
        self.failures = np.full(self.env.nA, self.prior_b, dtype=np.int32)
        self.means = self.successes/(self.successes + self.failures)
        self.ucl_bonus = beta.ppf(1-1/(self.K*(self.t+1)), self.successes, self.failures)
        
    def get_action(self, obs):
        self.arm_vals = self.means + self.ucl_bonus
        act_dist = softmax(self.arm_vals/self.softmax_temp)
        act = self.np_random.choice(range(self.env.nA), p=act_dist)
        return act
    
    def learn(self, old_obs, act, rew, new_obs, done):
        self.t += 1
        if rew > 0.5: 
            self.successes[act] += 1
        else:
            self.failures[act] += 1
        self.means[act] = self.successes[act]/(self.successes[act] + self.failures[act])

        #update UCL bonus
        self.ucl_bonus = beta.ppf(1-1/(self.K*(self.t+1)), self.successes, self.failures)


    def get_state(self):
        state = {}
        state['t'] = self.t
        state['K'] = self.K
        state['softmax_temp'] = self.softmax_temp
        state['successes'] = self.successes
        state['failures'] = self.failures
        state['means'] = self.means
        state['ucl_bonus'] = self.ucl_bonus
        return state

    #For MCMC-based methods
    def get_initial_state(self, **kwargs):
        state = {}
        state['t'] = 0
        state['K'] = self.K
        state['softmax_temp'] = self.softmax_temp
        state['successes'] = np.full(self.env.nA, self.prior_a, dtype=np.int32)
        state['failures'] = np.full(self.env.nA, self.prior_b, dtype=np.int32)
        state['means'] = state['successes']/(state['successes'] + state['failures'])
        lookup = (state['K'], state['t'], self.prior_a, self.prior_b)
        if lookup not in UCLBetaBernoulliBanditPolicy.PPF_cache:
            UCLBetaBernoulliBanditPolicy.PPF_cache[lookup] = beta.ppf(1-1/(state['K']*(state['t']+1)), self.prior_a, self.prior_b)
        state['ucl_bonus'] = np.array([UCLBetaBernoulliBanditPolicy.PPF_cache[lookup]]*self.env.nA)
        return state

    def get_action_from_state(self, state, obs):
        arm_vals = state['means'] + state['ucl_bonus']
        act_dist = softmax(arm_vals/state['softmax_temp'])
        act = self.np_random.choice(range(self.env.nA), p=act_dist)
        return act

    def update_state(self, state, old_obs, act, rew, new_obs):
        state['t'] += 1
        if rew > 0.5:
            state['successes'][act] += 1
        else:
            state['failures'][act] += 1
        state['means'][act] = state['successes'][act]/(state['successes'][act] + state['failures'][act]) 

        #update UCL bonus 
        for a in range(self.env.nA):
            lookup = (state['K'], state['t'], state['successes'][a], state['failures'][a])
            if lookup not in UCLBetaBernoulliBanditPolicy.PPF_cache:
                UCLBetaBernoulliBanditPolicy.PPF_cache[lookup] = beta.ppf(1-1/(state['K']*(state['t']+1)), state['successes'][a], state['failures'][a])
            state['ucl_bonus'][a] = UCLBetaBernoulliBanditPolicy.PPF_cache[lookup]
        return state
    
    def likelihood(self, state, obs, act):
        arm_vals = state['means'] + state['ucl_bonus']
        act_dist = softmax(arm_vals/state['softmax_temp'])
        return act_dist[act]

    def log_likelihood(self, state, obs, act):
        return np.log(self.likelihood(state, obs, act))


if __name__ == "__main__":
    from assistive_bandits.envs import BanditEnv
    bandit_env = BanditEnv()
    pi_H = DeterministicUCLBetaBernoulliBanditPolicy(bandit_env)
    print(bandit_env.theta)

    ob = 0 

    for t in range(50):
        
        act = pi_H.get_action(ob)
        print(act, pi_H.arm_vals)
        
        ob, rew, done, info = bandit_env.step(act)
        print(rew)
        
        pi_H.learn(0, act, rew, ob, done)
        if done:
            break
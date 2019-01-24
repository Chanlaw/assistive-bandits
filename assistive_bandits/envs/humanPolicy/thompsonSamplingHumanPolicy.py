from __future__ import print_function, division

import numpy as np
import tensorflow as tf

from sandbox.rocky.tf.spaces.discrete import Discrete #supercedes the gym spaces
from assistive_bandits.envs.utils import softmax
from assistive_bandits.envs.humanPolicy.humanPolicy import HumanPolicy
from scipy.stats import beta

class TSBetaBernoulliBanditPolicy(HumanPolicy):
    """ 
    Implements a Thompson sampling-based policy for a Beta-Bernoulli 
Bandit.
    """
    
    def __init__(self, env):
        super(TSBetaBernoulliBanditPolicy, self).__init__(env)
    
    def reset(self):
        self.successes = np.ones(self.env.nA)
        self.failures = np.ones(self.env.nA)
        self.means = self.successes/(self.successes + self.failures)
        
    def get_action(self, obs):
        self.sample_means = self.np_random.beta(self.successes, self.failures)
        return np.argmax(self.sample_means)
            
    def learn(self, old_obs, act, rew, new_obs, done):
        if rew > 0.5: 
            self.successes[act] += 1
        else:
            self.failures[act] += 1
        self.means[act] = self.successes[act]/(self.successes[act] + 
self.failures[act])

    def get_initial_state(self, **kwargs):
        successes = np.ones(self.env.nA, dtype=np.float32)
        failures = np.ones(self.env.nA, dtype=np.float32)
        return {'successes': successes, 'failures':failures}

    def update_state(self, state, old_obs, act, rew, new_obs):
        if rew > 0.5: 
            state['successes'][act] += 1
        else:
            state['failures'][act] += 1
        return state

    def get_action_from_state(self, state, obs):
        sample_means = self.np_random.beta(state['successes'], state['failures'])
        return np.argmax(sample_means)

    def get_particle_from_state(self, state, obs):
        """
        Returns a particle from this state, as well as the log_density of this particle
        """
        sample_means = beta.rvs(state['successes'], state['failures'])
        log_density = np.sum(beta.logpdf(sample_means, state['successes'], state['failures']))
        return sample_means, log_density

    def get_action_from_particle(self, sample_means):
        return np.argmax(sample_means)

class AnnealedTSBBBPolicy(TSBetaBernoulliBanditPolicy):
    """
    Implements an annealed version of the Thompson sampling-based policy for a 
    Beta-Bernoulli Bandit. Instead of sampling one particle, we can now sample many. 
    """

    def __init__(self, env, n_particles=2):            
        self.n_particles = n_particles
        super(AnnealedTSBBBPolicy, self).__init__(env)

    def get_action(self, obs):
        self.sample_means = self.np_random.beta([self.successes]*self.n_particles,
                                                 [self.failures]*self.n_particles)
        self.sample_means = np.mean(self.sample_means, axis=0)
        return np.argmax(self.sample_means)

    def get_action_from_state(self, state, obs):
        sample_means = self.np_random.beta(state['successes']*self.n_particles, 
                                            state['failures']*self.n_particles)
        self.sample_means = np.mean(self.sample_means, axis=0)
        return np.argmax(sample_means)

    def get_particle_from_state(self, state, obs):
        """
        Returns a particle from this state, as well as the log_density of this particle
        """
        sample_means = beta.rvs(state['successes']*self.n_particles, 
                                state['failures']*self.n_particles)
        log_density = np.sum(beta.logpdf(sample_means, state['successes']*self.n_particles, 
                                state['failures']*self.n_particles))
        return sample_means, log_density

    def get_action_from_particle(self, sample_means):
        return np.argmax(np.mean(sample_means, axis=0))


class InfAnnealedTSBBBPolicy(TSBetaBernoulliBanditPolicy):
    """
    Implements the infinitely annealed version of the Thompson sampling-based policy.
    Instead of sampling one particle, we just use the mean of the posterior distribution.
    """

    def __init__(self, env):
        super(InfAnnealedTSBBBPolicy, self).__init__(env)

    def get_action(self, obs):
        self.sample_means = self.successes/(self.successes+self.failures)
        return self.np_random.choice(np.where(np.isclose(self.sample_means, self.sample_means.max()))[0])

    def get_action_from_state(self, state, obs):
        sample_means = state['successes']/(state['successes']+state['failures'])
        return self.np_random.choice(np.where(np.isclose(sample_means, sample_means.max()))[0])

    def get_particle_from_state(self, state, obs):
        """
        Returns a particle from this state, as well as the log_density of this particle
        """
        sample_means = state['successes']/(state['successes']+state['failures'])
        log_density = 0
        return sample_means, log_density

    def get_action_from_particle(self, sample_means):
        return np.argmax(sample_means)


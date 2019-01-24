from __future__ import print_function, division

import sys
from itertools import cycle

import numpy as np
from scipy.stats import bernoulli, uniform #default distributions

from gym import Env, spaces
from gym.utils import seeding
from gym.envs.toy_text import discrete

from rllab.envs.base import EnvSpec
from sandbox.rocky.tf.spaces.discrete import Discrete #supercedes the gym space

class BanditEnv(discrete.DiscreteEnv):
    """
    A general (stochastic) multi-armed bandit environment. Contextual support to follow. 
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, n_arms=4, reward_dist=None, reward_args_generator=None,
                 independent_arms=True, horizon=15):
        """
        Initialize a bandit environment. 
        
        Args:
            n_arms (int): number of arms in this bandit. 
            reward_dist: a generator for a discrete scipy.stats.rv_discrete used to sample from the arms. 
                    If independent_arms, then this should return a single real random variable.
                    Else, then this should return a vector-valued random variable.
            reward_args_generator (iterable): a generator of length n_arms for reward arguments. 
                                        Will be used to create args for reward_dist.
                                        Cannot be None if reward_dist is not None.
                                        Needs to continue on forever. 
            independent_arms (bool): If True, sample each arm's reward independently
                                (conditioned on reward_args_generator). 
                            If False, generate a simple sample from reward_dist.
            horizon (int): the number of timesteps to run this environment for. 
            
        """
        if not independent_arms:
            raise NotImplementedError("Only conditionally dependent arms are currently supported.")
        
        if reward_dist is None:
            reward_dist = bernoulli
            reward_args_generator = ({'p': uniform.rvs()} for _ in cycle([0]))
        elif reward_args_generator is None:
            raise ValueError("reward_args_generator cannot be None if reward_generator is specified.")
            
        self.n_arms = n_arms
        self.reward_dist = reward_dist
        self.reward_args_generator = reward_args_generator
        self.horizon = horizon
            
        self.steps_taken = 0
        
        P = {0:{}}
        for i in range(n_arms):
            P[0][i] = [(0.5, 0, 1.0, False), (0.5, 0, 0.0, False)]
        isd = [1.0]
        
        super(BanditEnv, self).__init__(1, n_arms, P, isd)
        
        self.action_space = Discrete(self.nA)
        self.observation_space = Discrete(self.nS)
        
    def _reset(self):
        """ Resets the environment. Samples new arm distributions using reward_dist"""
        self.arms = []
        self.theta = []
        self.arm_means = []
        for _, args in zip(range(self.n_arms), self.reward_args_generator):
            arm = self.reward_dist(**args)
            self.arms.append(arm)
            self.theta.append(args)
            self.arm_means.append(arm.mean())
        #reset P
        self.P = {0:{}}
        for i in range(self.n_arms):
            self.P[0][i] = [(self.arms[i].pmf(x), 0, x, False) for x in range(self.arms[i].a, self.arms[i].b+1)]
        self.steps_taken = 0
        self.lastreward = 0

        return super(BanditEnv, self)._reset()
    
    def _step(self, a):
        """
        Take a step in the environment.
        
        Args:
            a (int): integer between 0 and n_arms. Denotes which arm to pull. 
            
        Returns:
            (tuple): a tuple consisting of (obs, rew, done, info)
        """
        self.steps_taken += 1
        self.lastaction = a
        r = self.arms[a].rvs()
        self.lastreward = r
        d = (self.steps_taken) >= self.horizon
        return (self.s, r, d, {"last action": a, "theta": self.theta, "arm_means": self.arm_means})
    
    def _render(self, mode='human', close=False):
        """
        'Renders' the environment by printing out information about it. 
        Args:
            mode (string): does nothing. currently only prints human readable thingies. 
            close: whether to close the renderer. 
        """
        if close:
            return
        outfile = sys.stdout
        outfile.write("Step {} of {}:\n".format(self.steps_taken, self.horizon))
        outfile.write("Theta: {}\n".format(self.theta))
        if self.lastaction is not None:
            outfile.write("Last action: {}, Last reward: {}\n" \
                                .format(self.lastaction,self.lastreward))
        else:
            outfile.write("\n")

    @property
    def spec(self):
        return EnvSpec(
            observation_space=self.observation_space,
            action_space=self.action_space,
        )
    
    @spec.setter
    def spec(self, value):
        if value is not None:
            spec = value
    
    def log_diagnostics(self, paths, *args, **kwargs):
        return None
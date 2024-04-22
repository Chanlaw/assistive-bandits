from __future__ import print_function, division

import numpy as np 

# from gym import Env
from gym.utils import seeding
from gym.envs.toy_text import discrete

from rllab.envs.base import Env, EnvSpec
from sandbox.rocky.tf.spaces.box import Box #supercedes the gym space
from sandbox.rocky.tf.spaces.discrete import Discrete #supercedes the gym space
from crl.spaces.multi_discrete import MultiDiscrete #supercedes the gym space

class FeatureMDP(discrete.DiscreteEnv):
    """
    Generalized featureMDP, built on top of Gym's DiscreteEnv. 

    Each state has features associated with it, and reward is a linear function of features.

    Args:
        T (dict): a dictionary of dictionary of lists, where:
            T[s][a] == [(probability, nextstate, done)]
        theta (np.array): a 1-d array parameterizing the reward function
        isd (list): initial state distribution. Should be an array of length nS
        feats (list): a list of features (1-d arrays) associated with the state. 
    """

    def __init__(self, nS, nA, T, theta, isd, feats):
        self.n_feats = len(feats[0])

        P = {}
        for s in T:
            P[s] = {}
            for a in T[s]:
                P[s][a] = []
                for prob, next_s, done in T[s][a]:
                    rew = np.dot(feats[s], theta)
                    P[s][a].append( (prob, next_s, rew, done))
        self.T = T
        self.feats = feats
        self.theta = theta
        

        super(FeatureMDP, self).__init__(nS, nA, P, isd)

        self.action_space = Discrete(self.nA)
        self.observation_space = Discrete(self.nS)
        

    def get_features(self, state):
        return self.feats[state]

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


class ToyMDP(FeatureMDP):
    """ a simple MDP with 3 states and 2 features per state 

             S_1
            [1, 1]
           /
        S_0 - [2, 0]
           \
            [0, 1]
             S_2
    """
    def __init__(self, theta=np.array([0.4, 0.8]), random_theta=False, horizon=3):
        if theta is None and not random_theta:
            raise ValueError("theta can't be none if random_theta is false")
        self.random_theta = random_theta
        self.horizon = horizon
        feats = [np.array([2,0]), np.array([1,1]), np.array([0,1])]
        isd = [1.0, 0, 0]
        T = {0: {}, 1:{}, 2:{}}
        for i in range(3):
            T[0][i] = [(1.0, i, False)]
            T[1][i] = [(1.0, 1, False)]
            T[2][i] = [(1.0, 2, False)]
        super(ToyMDP, self).__init__(3, 3, T, theta, isd, feats)

    def _reset(self):
        self.n_timesteps = 0
        if self.random_theta:
            theta = np.random.rand(2)*2 - 1
            P = {}
            for s in self.T:
                P[s] = {}
                for a in self.T[s]:
                    prob, next_s, done = self.T[s][a]
                    rew = np.dot(feats[s], theta)
                    P[s][a] = (prob, next_s, rew, done)
            self.P = P
        return super(ToyMDP, self)._reset()

    def _step(self, a):
        s, r, d, info = super(ToyMDP, self)._step(a)
        self.n_timesteps += 1
        d = (self.n_timesteps >= self.horizon)
        return (s,r,d,info)


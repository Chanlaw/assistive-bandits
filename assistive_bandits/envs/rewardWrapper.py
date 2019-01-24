from __future__ import print_function, division

import numpy as np 

import gym
from gym import Env, spaces
from gym.utils import seeding
from gym.envs.toy_text import discrete

from rllab.misc import special
from rllab.envs.base import EnvSpec
from sandbox.rocky.tf.spaces.discrete import Discrete #supercedes the gym space
from assistive_bandits.spaces.multi_discrete import MultiDiscrete #supercedes the gym space
from assistive_bandits.spaces.box import Box #supercedes the gym space

class DiscreteRewardWrapperEnv(Env):
    """
    Wrapper env that makes the reward part of the state.
    """
    def __init__(self, env, min_reward=0., max_reward=1.):
        """
        """
        assert(isinstance(env.observation_space, MultiDiscrete))
        self.nA = env.nA
        self.wrapped_env = env
        self.action_space = self.wrapped_env.action_space
        low = np.append(env.observation_space.low, [min_reward])
        high = np.append(env.observation_space.high, [max_reward])
        self.observation_space = MultiDiscrete([[l,h] for l, h in zip(low, high)])
        
        self._seed()
        self._reset()
        
    def _seed(self, seed=None):
        """ 
        Sets the random seed for both the wrapped environment and random choices made here.

        Args:
            seed (int): the seed to use. 

        returns:
            (list): list containing the random seed. 
        """
        self.np_random, seed = seeding.np_random(seed)
        self.wrapped_env.seed(seed)
        return [seed]
    
    def _reset(self):
        """ 
        Resets the environment. 

        Returns:
            (int): the current state of the environment.
        """
        obs = self.wrapped_env.reset()
        self.done = False
        self.last_rew = 0.0
        self.accumulated_rew = 0.0
        obs = np.append(obs, 0)
        return obs
    
    def _step(self, a):
        """"""
        obs, rew, done, info = self.wrapped_env.step(a)
        self.done = done
        self.last_rew = rew
        self.accumulated_rew += rew
        obs = np.append(obs, rew)
        info["accumulated rewards"] = self.accumulated_rew
        return (obs, rew, done, info)
    
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

    
class RewardWrapperEnv(Env):
    """
    Wrapper env that makes the reward part of the state.
    """
    def __init__(self, env, min_reward=0., max_reward=1.):
        """
        """
        self.nA = env.nA
        self.wrapped_env = env
        self.action_space = self.wrapped_env.action_space
        if isinstance(env.observation_space, Discrete) or isinstance(env.observation_space, MultiDiscrete):
            low = np.array([0.] * env.observation_space.n + [min_reward])
            high = np.array([1.] * env.observation_space.n + [max_reward])
        else:
            low = np.append(env.observation_space.low, [min_reward])
            high = np.append(env.observation_space.high, [max_reward])
        self.observation_space = Box(low, high)
        
        self._seed()
        self._reset()
        
    def _seed(self, seed=None):
        """ 
        Sets the random seed for both the wrapped environment and random choices made here.

        Args:
            seed (int): the seed to use. 

        returns:
            (list): list containing the random seed. 
        """
        self.np_random, seed = seeding.np_random(seed)
        self.wrapped_env.seed(seed)
        return [seed]
    
    def _reset(self):
        """ 
        Resets the environment. 

        Returns:
            (int): the current state of the environment.
        """
        obs = self.wrapped_env.reset()
        self.done = False
        self.last_rew = 0.0
        self.accumulated_rew = 0.0
        flat_obs = self.wrapped_env.observation_space.flatten(obs)
        flat_obs = np.append(flat_obs, 0.)
        return flat_obs
    
    def _step(self, a):
        """"""
        obs, rew, done, info = self.wrapped_env.step(a)
        self.done = done
        self.last_rew = rew
        self.accumulated_rew += rew
        flat_obs = self.wrapped_env.observation_space.flatten(obs)
        flat_obs = np.append(flat_obs, rew)
        info["accumulated rewards"] = self.accumulated_rew
        return (flat_obs, rew, done, info)
    
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
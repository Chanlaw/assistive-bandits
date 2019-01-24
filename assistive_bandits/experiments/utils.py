from __future__ import print_function, division

import math
import numpy as np
import tensorflow as tf
import gym

def categorical_sample(prob_n):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np.random.rand()).argmax()

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def convert_teleop_rllab_to_traj_format(paths):
    """
    Converts the collected data from an rllab sampler for the teleop humanwrapper environment
    to the trajectory format defined in assistive_bandits.envs.HumanPolicy.rolloutUtils.py.
    """
    trajs = []
    for p in paths:
        traj = {}
        traj['length'] = len(p['rewards'])
        obs_R = env.observation_space.unflatten_n(p['observations'])
        traj['obs'] = obs_R[:,0]
        traj['acts'] = obs_R[:,1]
        traj['rews'] = p['rewards']
        traj['best_arm'] = np.argmax(p['env_infos']['arm_means'][0])
        traj['total_rew'] = sum(p['rewards'])
        trajs.append(traj)
    
    return trajs
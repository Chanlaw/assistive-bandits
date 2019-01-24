from __future__ import print_function, division

import numpy as np
import tensorflow as tf

import gym
from gym.utils import seeding
from gym import spaces

from sandbox.rocky.tf.spaces.discrete import Discrete #supercedes the gym spaces
from assistive_bandits.envs.utils import softmax

class HumanPolicy:
    """Abstract class for human policies"""
    def __init__(self, env):
        self.env = env
        self.seed()
        self.reset()
        
    def get_action(self, obs):
        raise NotImplementedError
    
    def learn(self, old_obs, act, rew, new_obs, done):
        raise NotImplementedError
    
    def reset(self):
        raise NotImplementedError
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def get_state(self):
        """ Returns a dict of the human's internal state """
        raise NotImplementedError
        
    def get_flat_state(self):
        """ returns a numpy vector """
        raise NotImplementedError
    
    # Functions for MCMC
    def get_initial_state(self):
        """ Returns a dict of the human's initial internal state """
        raise NotImplementedError

    def get_action_from_state(self, state, obs):
        raise NotImplementedError

    def update_state(self, state, old_obs, act, rew, new_obs):
        raise NotImplementedError

    def log_likelihood(self, state, obs, act):
        """ Computes the log likelihood of taking the action given the state. """
        raise NotImplementedError
    
    def act_probs_from_counts(self, counts, **kwargs):
        """ Returns the act probabilities given the current counts. """
        raise NotImplementedError
        
class RandomPolicy(HumanPolicy):
    def __init__(self, env):
        self.env = env
        self.seed()
        self.reset()
        
    def get_action(self, obs):
        return env.action_space.sample()
    
    def learn(self, old_obs, act, rew, new_obs):
        return
    
    def reset(self):
        return
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def get_state(self):
        return {}
    
    def get_initial_state(self, **kwargs):
        return {}

    def get_action_from_state(self, state, obs):
        return np.random.choice(range(self.env.nA))

    def update_state(self, state, old_obs, act, rew, new_obs):
        return {}

    def log_likelihood(self, state, obs, act):
        return np.log(self.likelihood(state, obs, act))
        
    def likelihood(self, state, obs, act):
        """ Computes the likelihood of taking the action given the state. """
        return 1.0/self.env.n_arms

    def act_probs_from_counts(self, counts, **kwargs):
        return [1.0/self.env.n_arms for _ in range(self.env.n_arms)]

class QLearningBanditPolicy(HumanPolicy):
    """
    Tabular Q-learning policy for bandit problems with softmax exploration.
    """
    def __init__(self, env, softmax_temp=0.2, alpha=0.5, initial_q_value=0., discount=0.99):
        assert (isinstance(env.observation_space, Discrete) \
                or isinstance(env.observation_space, spaces.Discrete))
        assert (isinstance(env.action_space, Discrete) \
                or isinstance(env.action_space, spaces.Discrete))
        
        self.softmax_temp=softmax_temp
        self.alpha=alpha
        self.initial_q_value=initial_q_value
        self.discount=discount
        
        super(QLearningBanditPolicy, self).__init__(env)
    
    def reset(self):
        #reset Q Table
        self.Q_human = np.full((self.env.nS, self.env.nA), 
                                 self.initial_q_value, dtype=np.float32)
        
    def get_action(self, obs):
        act_dist = softmax(self.Q_human[obs]/self.softmax_temp)
        act = self.np_random.choice(range(self.env.nA), p=act_dist)
        return act
    
    def learn(self, old_obs, act, rew, new_obs, done):
        Q_next = np.max(self.Q_human[new_obs])
        Q_sa = self.Q_human[old_obs][act]
        self.Q_human[old_obs][act] = (self.alpha * (rew + self.discount*Q_next - Q_sa) + Q_sa)

    def get_state(self):
        return {'Q_human': self.Q_human}
    
    #For MCMC-based methods
    def get_initial_state(self, **kwargs):
        Q_human = np.full((self.env.nS, self.env.nA), 
                                 self.initial_q_value, dtype=np.float32)
        return {'Q_human': Q_human}

    def get_action_from_state(self, state, obs):
        act_dist = softmax(state['Q_human'][obs]/self.softmax_temp)
        act = np.random.choice(range(self.env.nA), p=act_dist)
        return act

    def update_state(self, state, old_obs, act, rew, new_obs):
        Q_human = state['Q_human'].copy()
        Q_next = np.max(Q_human[new_obs])
        Q_sa = Q_human[old_obs][act]
        Q_human[old_obs][act] = (self.alpha * (rew + self.discount*Q_next - Q_sa) + Q_sa)
        return {'Q_human': Q_human}
    
    def likelihood(self, state, obs, act):
        """ Computes the likelihood of taking the action given the state. """
        return softmax(state['Q_human'][obs]/self.softmax_temp)[act]
    
    def log_likelihood(self, state, obs, act):
        return np.log(self.likelihood(state, obs, act))

class SoftmaxBanditPolicy(HumanPolicy):
    """
    Tabular Q-learning policy for bandit problems with softmax exploration.

    Uses empirical means instead of Q table. 

    THIS IS NOT COMPLETE. DO NOT USE. 
    """
    def __init__(self, env, softmax_temp=0.2):
        assert (isinstance(env.observation_space, Discrete) \
                or isinstance(env.observation_space, spaces.Discrete))
        assert (isinstance(env.action_space, Discrete) \
                or isinstance(env.action_space, spaces.Discrete))
        
        self.softmax_temp=softmax_temp
        self.alpha=alpha
        self.initial_q_value=initial_q_value
        self.discount=discount
        
        super(SoftmaxBanditPolicy, self).__init__(env)
    
    def reset(self):
        #reset Q Table
        self.Q_human = np.full((self.env.nS, self.env.nA), 
                                 self.initial_q_value, dtype=np.float32)
        
    def get_action(self, obs):
        act_dist = softmax(self.Q_human[obs]/self.softmax_temp)
        act = self.np_random.choice(range(self.env.nA), p=act_dist)
        return act
    
    def learn(self, old_obs, act, rew, new_obs, done):
        Q_next = np.max(self.Q_human[new_obs])
        Q_sa = self.Q_human[old_obs][act]
        self.Q_human[old_obs][act] = (self.alpha * (rew + self.discount*Q_next - Q_sa) + Q_sa)

    def get_state(self):
        return {'Q_human': self.Q_human}
    
    #For MCMC-based methods
    def get_initial_state(self, **kwargs):
        Q_human = np.full((self.env.nS, self.env.nA), 
                                 self.initial_q_value, dtype=np.float32)
        return {'Q_human': Q_human}

    def get_action_from_state(self, state, obs):
        act_dist = softmax(state['Q_human'][obs]/self.softmax_temp)
        act = np.random.choice(range(self.env.nA), p=act_dist)
        return act

    def update_state(self, state, old_obs, act, rew, new_obs):
        Q_human = state['Q_human'].copy()
        Q_next = np.max(Q_human[new_obs])
        Q_sa = Q_human[old_obs][act]
        Q_human[old_obs][act] = (self.alpha * (rew + self.discount*Q_next - Q_sa) + Q_sa)
        return {'Q_human': Q_human}
    
    def likelihood(self, state, obs, act):
        """ Computes the likelihood of taking the action given the state. """
        return softmax(state['Q_human'][obs]/self.softmax_temp)[act]
    
    def log_likelihood(self, state, obs, act):
        return np.log(self.likelihood(state, obs, act))

class EpsGreedyBanditPolicy(HumanPolicy):
    """
    Epsilon-greedy policy for bandit problems. Maintain the empirical means of each of the arms, 
        then choose the arm with highest mean with probability 1-epsilon and a random arm with 
        probability epsilon.
    """
    def __init__(self, env, epsilon = 0.1):
        assert (isinstance(env.observation_space, Discrete) \
                or isinstance(env.observation_space, spaces.Discrete))
        assert (isinstance(env.action_space, Discrete) \
                or isinstance(env.action_space, spaces.Discrete))
        
        self.epsilon = epsilon
        
        super(EpsGreedyBanditPolicy, self).__init__(env)
        
    def reset(self):
        #reset empirical mean table
        self.means = np.zeros((self.env.observation_space.n, self.env.action_space.n),
                              dtype=np.float32)
        self.counts = np.zeros((self.env.observation_space.n, self.env.action_space.n),
                               dtype=np.int32)
        
    def get_action(self, obs):
        act = self.env.action_space.sample() if self.np_random.random_sample() < self.epsilon \
                            else self.np_random.choice(np.where(np.isclose(self.means[obs],
                                                                  self.means[obs].max()))[0])
        return act
    
    def learn(self, old_obs, act, rew, new_obs, done):
        count = self.counts[old_obs][act] = self.counts[old_obs][act] + 1
        self.means[old_obs][act] = (count-1)*self.means[old_obs][act]/count + rew/count
        
    def get_state(self):
        return {'means': self.means, 'counts': self.counts}
    
    #Functions for MCMC
    def get_initial_state(self, **kwargs):
        means = np.zeros((self.env.observation_space.n, self.env.action_space.n),
                              dtype=np.float32)
        counts = np.zeros((self.env.observation_space.n, self.env.action_space.n),
                               dtype=np.int32)
        return {'means': means, 'counts': counts}

    def get_action_from_state(self, state, obs):
        act = self.np_random.choice(range(self.env.nA))if np.random.random_sample() < self.epsilon \
                            else self.np_random.choice(np.where(np.isclose(state['means'][obs],
                                                                  state['means'][obs].max()))[0])
        return act

    def update_state(self, state, old_obs, act, rew, new_obs):
        means = state['means'].copy()
        counts = state['counts'].copy()
        count = counts[old_obs][act] = counts[old_obs][act] + 1
        means[old_obs][act] = (count-1)*means[old_obs][act]/count + rew/count
        return {'means': means, 'counts': counts}
    
    def likelihood(self, state, obs, act):
        """ Computes the likelihood of taking the action given the state. """
        greedy_arms = np.where(np.isclose(state['means'][obs], state['means'][obs].max()))[0]
        return (1-self.epsilon)/len(greedy_arms)+ self.epsilon/self.env.n_arms \
                    if act in greedy_arms else  self.epsilon/self.env.n_arms
    
    def log_likelihood(self, state, obs, act):
        return np.log(self.likelihood(state, obs, act))
    
    # Used for RTDP
    def act_probs_from_counts(self, counts, **kwargs):
        means = np.array([(counts[2*i]-1)/(counts[2*i] + counts[2*i+1]-2) if counts[2*i] + counts[2*i+1]-2 > 0 else 0 
                          for i in range(self.env.n_arms)])
        act_probs = np.full(self.env.n_arms, self.epsilon/self.env.n_arms)
        greedy_arms = np.where(np.isclose(means,means.max()))[0]
        act_probs[greedy_arms] += (1-self.epsilon)/len(greedy_arms)
        return act_probs


class GreedyBanditPolicy(EpsGreedyBanditPolicy):
    """
    Greedy policy for bandit problems. Just a eps-greedy policy with epsilon set to zero.
    """
    def __init__(self, env):
        super(GreedyBanditPolicy, self).__init__(env, epsilon=0)


class WSLSBanditPolicy(HumanPolicy):
    """
    Win-stay-lose-shift policy. If the arm's reward is higher than average, stay. 
        If the arm's reward is lower than average, switch to a random other arm.
    """
    def __init__(self, env):
        assert (isinstance(env.observation_space, Discrete) \
                or isinstance(env.observation_space, spaces.Discrete))
        assert (isinstance(env.action_space, Discrete) \
                or isinstance(env.action_space, spaces.Discrete))
        super(WSLSBanditPolicy, self).__init__(env)
    
    def reset(self):
        self.curr_arm = 0
        self.mean = 0.5
        self.count = 8
    
    def get_action(self, obs):
        return self.curr_arm
    
    def learn(self, old_obs, act, rew, new_obs, done):
        if rew < self.mean: #select another arm at random
            arm = self.np_random.randint(0, self.env.nA-1)
            if arm >= self.curr_arm:
                arm += 1
            self.curr_arm = arm
        else:
            self.curr_arm = act
        count = self.count = self.count + 1
        self.mean = (count - 1) * self.mean / count + rew /count
        
    def get_state(self):
        return {'curr_arm': self.curr_arm, 'mean': self.mean, 'count': self.count}
    
    def get_initial_state(self, **kwargs):
        curr_arm = 0
        mean = 0.5
        count = 8
        return {'curr_arm': curr_arm, 'mean': mean, 'count': count}

    def update_state(self, state, old_obs, act, rew, new_obs):
        mean = state['mean']
        curr_arm = state['curr_arm']
        if rew < mean: #select another arm at random
            arm = np.random.randint(0, self.env.nA-1)
            if arm >= curr_arm:
                arm += 1
            curr_arm = arm
        else:
            curr_arm = act
        count = state['count'] = state['count']+1
        mean = (count - 1) * mean / count + rew /count
        return {'curr_arm': curr_arm, 'mean': mean, 'count': count}
    
    def get_action_from_state(self, state, obs):
        return state['curr_arm']
    
    def likelihood(self, state, obs, act):
        return 1 if act == state['curr_arm'] else 1e-10
    
    def log_likelihood(self, state, obs, act):
        """ Computes the log likelihood of taking the action given the state. """
        return 0 if act == state['curr_arm'] else -1e10
    
    def act_probs_from_counts(self, counts, last_arm, last_rew, **kwargs):
        successes = sum(counts[2*i] for i in range(self.env.n_arms))
        failures = sum(counts[2*i+1] for i in range(self.env.n_arms))
        mean = successes/(successes + failures)
        if last_rew > mean:
            act_probs = np.zeros(self.env.n_arms)
            act_probs[last_arm] = 1.
        else:
            act_probs = np.full(self.env.n_arms, 1/(self.env.n_arms -1))
            act_probs[last_arm] = 0.
        return act_probs


from __future__ import print_function, division

import time
import numpy as np
import copy

from assistive_bandits.envs import BanditEnv
from assistive_bandits.envs.rewardWrapper import DiscreteRewardWrapperEnv
from assistive_bandits.envs.humanWrapper import HumanCRLWrapper, HumanIterativeWrapper
from assistive_bandits.envs.humanPolicy import HumanPolicy, QLearningBanditPolicy, EpsGreedyBanditPolicy, WSLSBanditPolicy

from experiments.POMCP import Model, BeliefNode, ActionMap, BeliefTree
from experiments.POMCP.banditPOMCP import BanditModel
class BernoulliHumanBanditModel(Model):
    """ Black box simulator for humans playing bernoulli bandits, where reward is observed."""
    def __init__(self, env):
        assert (isinstance(env, DiscreteRewardWrapperEnv))
        self.env = env.wrapped_env
        self.bandit = env.wrapped_env.wrapped_env
    
    def generate_state_particles(self, n_particles=1):
        """
        Sample n state particles from the initial distribution. A particle is a dict containing a
            set of reward parameters for each arm and the human's internal state

        Args:
            n_particles (int): the number of particles to sample

        Returns:
            (list): the list of n particles. 
        """
        particles = []

        particle = {}
        particle['successes']= np.ones(self.bandit.n_arms, dtype=np.int32)
        particle['failures']= np.ones(self.bandit.n_arms, dtype=np.int32)
        particle['state_H'] = self.env.pi_H.get_initial_state()
        particle['steps_taken'] = 0

        particles.append(particle)

        return particles

    def sample_transition(self, state, act):
        """ 
        Creates a successor to the current state particle. 

        Args:
            state: the current state particle
            act (int): the action taken by the agent in the environment

        Returns:
            (tuple): tuple of (state', obs', rew, done)
        """
        if act < 0 or act >= self.env.nA:
            raise ValueError("{} is not a legal action".format(act))
        new_state = state.copy()
        new_state['steps_taken'] = state['steps_taken'] + 1
        new_state['successes'] = state['successes'].copy()
        new_state['failures'] = state['failures'].copy()
        if isinstance(self.env, HumanCRLWrapper):
            raise NotImplementedError
        elif isinstance(self.env, HumanIterativeWrapper):
            if new_state['steps_taken']%2 == 1: #robot acts
                a_H = self.env.nA
                rew = np.random.binomial(1, np.random.beta(state['successes'][act], state['failures'][act])) #sample reward from arm
                if rew > 0.5:
                    new_state['successes'][act] += 1
                else:
                    new_state['failures'][act] += 1
                done = new_state['steps_taken'] >= self.env.wrapped_env.horizon
                new_state['state_H'] = self.env.pi_H.update_state(state['state_H'], 0, act, rew, 0)
            else: #human acts
                a_H = self.env.pi_H.get_action_from_state(state['state_H'], 0)
                rew = np.random.binomial(1, np.random.beta(state['successes'][a_H], state['failures'][a_H])) #sample reward from arm
                if rew > 0.5:
                    new_state['successes'][act] += 1
                else:
                    new_state['failures'][act] += 1
                done = new_state['steps_taken'] >= self.env.wrapped_env.horizon
                new_state['state_H'] = self.env.pi_H.update_state(state['state_H'], 0, a_H, rew, 0)
        return (new_state, a_H*2+rew, rew, done)
    
class BetaBernoulliIterativeBanditPOUCTAgent(object):
    def __init__(self, env, pi_rollout=None, ucb_c=1, gamma=0.95, epsilon=0.01):
        """ Initializes this POMCP agent. """
        self.env = env
        self.pi_H = self.env.wrapped_env.pi_H
        self.gamma = gamma
        self.epsilon = epsilon
        self.timesteps = 0
        self.ucb_c = ucb_c

        self.model = BernoulliHumanBanditModel(env)
        self.tree = BeliefTree(self.model, env.nA)
        self.tree.root.state_particles = self.model.generate_state_particles()
        self.curr_node = self.tree.root
        
        self.last_act = None
        self.timesteps = 0
    
    def reset(self):
        self.timesteps = 0
        self.model = BernoulliHumanBanditModel(self.env)
        
        self.tree.reset()
        self.tree.root.state_particles = self.model.generate_state_particles()
        self.curr_node = self.tree.root
        
        self.last_act = None
        
    def get_action(self, obs, max_simulations=1000, max_time=1):
        """
        Performs simulations for max_time seconds or until max_simulations simulations have been
            completed, then chooses the action with the highest value. 
        """
        
        _, a_H, rew = obs
        obs = int(self.env.observation_space.to_discrete(obs))

        
        if self.timesteps > 0:
            #Update belief state based on last observation
            next_node, _ = self.curr_node.create_or_get_child(self.last_act, obs)
            t0 = time.perf_counter()
            state = self.curr_node.sample_particle()
            new_state = state.copy()
            new_state['steps_taken'] = state['steps_taken'] + 1
            new_state['successes'] = state['successes'].copy()
            new_state['failures'] = state['failures'].copy()
            
            if self.timesteps %2 ==1: #human acted last
                new_state['state_H'] = self.pi_H.update_state(state['state_H'], 0, self.last_act, rew, 0)
                if rew > 0.5:
                    new_state['successes'][self.last_act] += 1
                else:
                    new_state['failures'][self.last_act] += 1
            else:
                new_state['state_H'] = self.pi_H.update_state(state['state_H'], 0, a_H, rew, 0)
                if rew > 0.5:
                    new_state['successes'][a_H] += 1
                else:
                    new_state['failures'][a_H] += 1
            next_node.state_particles = [new_state]
                    
            self.curr_node = next_node
            self.tree.prune_siblings(self.curr_node)
        
        t0 = time.perf_counter()
        for i in range(max_simulations):
            state = self.curr_node.sample_particle()
            self.simulate(self.curr_node, state, 0)
            if time.perf_counter() - t0 > max_time:
                print('Simulations halted after {} simulations due to timeout.'.format(i+1))
                break
        # print('Choosing best action...')
        # print(self.curr_node.action_map.visit_counts)
        # print(self.curr_node.action_map.stats)
        self.last_act = act = np.argmax(self.curr_node.action_map.stats)
        self.timesteps += 1
        
        return act
            
    def simulate(self, bn, state, depth):
        """
        Recursively evaluates and updates the value of belief nodes starting from bn. 
        """
        if self.gamma ** depth < self.epsilon:
            return 0
        # Use UCB to choose the next action
        
        if bn.action_map.total_visit_count < self.env.nA:
            act = bn.action_map.total_visit_count
            new_state, obs, rew, done = self.model.sample_transition(state, act)
            if not done:
                val = rew + self.gamma * self.rollout(new_state, depth+1)
            else:
                val = rew
            bn.action_map.stats[act] = val
        else:
            log_N = np.log(bn.action_map.total_visit_count)
            ucb_vals = np.array(bn.action_map.stats) \
                        + self.ucb_c * np.sqrt([log_N/n for n in bn.action_map.visit_counts])
            act = np.argmax(ucb_vals)
            new_state, new_obs, rew, done = self.model.sample_transition(state, act)
            if not done:
                child_bn, is_new = bn.create_or_get_child(act, new_obs)
                if is_new:
                    val = rew + self.gamma * self.rollout(new_state, depth+1)
                else:
                    val = rew + self.gamma * self.simulate(child_bn, new_state, depth+1)
            else:
                val = rew
            count = bn.action_map.visit_counts[act]
            bn.action_map.stats[act] += (val - bn.action_map.stats[act])/count
            
        bn.action_map.visit_counts[act] += 1
        bn.action_map.total_visit_count += 1
        return val
                
    def rollout(self, state, depth):
        if self.gamma ** depth < self.epsilon:
            return 0
        
        # current rollout policy is just to pull the most pulled arm
        act = self.env.action_space.sample()
        new_state, obs, rew, done = self.model.sample_transition(state, act)
        if done:
            return rew
        return rew + self.gamma * self.rollout(new_state, depth+1)
    
if __name__ == "__main__":
    bandit = BanditEnv(n_arms=4)
    pi_H = EpsGreedyBanditPolicy(bandit)
    env = DiscreteRewardWrapperEnv(HumanIterativeWrapper(bandit, pi_H))
    agent = BetaBernoulliIterativeBanditPOUCTAgent(env)
    print(agent.get_action([0, 4, 1]))
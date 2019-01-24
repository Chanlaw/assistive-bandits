from __future__ import print_function, division

import time
import numpy as np
import copy

from assistive_bandits.envs import BanditEnv
from assistive_bandits.envs.rewardWrapper import DiscreteRewardWrapperEnv
from assistive_bandits.envs.humanWrapper import HumanCRLWrapper, HumanIterativeWrapper
from assistive_bandits.envs.humanPolicy import HumanPolicy, EpsGreedyBanditPolicy, WSLSBanditPolicy

from experiments.POMCP import Model, BeliefNode, ActionMap, BeliefTree

class BanditModel(Model):
    """ Black box simulator for a bandit problem. """
    def __init__(self, env):
        assert (isinstance(env, BanditEnv))
        self.bandit = env
        
    def generate_state_particles(self, n_particles):
        """
        Sample n state particles from the initial distribution. A particle is a dict containing a
            set of reward parameters for each arm.

        Args:
            n_particles (int): the number of particles to sample

        Returns:
            (list): the list of n particles. 
        """
        particles = []

        for i in range(n_particles):
            particle = {}
            arms = []
            theta = []
            for _, args in zip(range(self.bandit.n_arms), \
                               self.bandit.reward_args_generator):
                arms.append(self.bandit.reward_dist(**args))
                theta.append(args)
            particle['arms']= arms
            particle['theta'] = theta
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
        rew = new_state['arms'][act].rvs() #sample reward from arm
        done = new_state['steps_taken'] >= self.env.wrapped_env.horizon
        
        return (new_state, 0, rew, done)
        
class HumanBanditModel(Model):
    """ Black box simulator for humans playing bandits """
    def __init__(self, env):
        assert (isinstance(env, HumanCRLWrapper) or isinstance(env, HumanIterativeWrapper))
        self.env = env
        self.bandit = env.wrapped_env
    
    def generate_state_particles(self, n_particles):
        """
        Sample n state particles from the initial distribution. A particle is a dict containing a
            set of reward parameters for each arm and the human's internal state

        Args:
            n_particles (int): the number of particles to sample

        Returns:
            (list): the list of n particles. 
        """
        particles = []

        for i in range(n_particles):
            particle = {}
            arms = []
            theta = []
            for _, args in zip(range(self.bandit.n_arms), \
                               self.bandit.reward_args_generator):
                arms.append(self.bandit.reward_dist(**args))
                theta.append(args)
            particle['arms']= arms
            particle['theta'] = theta
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
        if isinstance(self.env, HumanCRLWrapper):
            if act == self.env.nA -1:
                a_H = self.env.pi_H.get_action_from_state(state['state_H'], 0)
                rew = new_state['arms'][a_H].rvs() - self.env.penalty #sample reward from arm
                done = new_state['steps_taken'] >= self.env.wrapped_env.horizon
                new_state['state_H'] = self.env.pi_H.update_state(new_state['state_H'], 0, a_H, rew, 0)
            else:
                a_H = self.env.nA - 1
                rew = new_state['arms'][act].rvs() #sample reward from arm
                done = new_state['steps_taken'] >= self.env.wrapped_env.horizon
                new_state['state_H'] = self.env.pi_H.update_state(new_state['state_H'], 0, act, rew, 0)
        elif isinstance(self.env, HumanIterativeWrapper):
            if new_state['steps_taken']%2 == 1: #robot acts
                a_H = self.env.nA
                rew = state['arms'][act].rvs() #sample reward from arm
                done = new_state['steps_taken'] >= self.env.wrapped_env.horizon
                new_state['state_H'] = self.env.pi_H.update_state(new_state['state_H'], 0, act, rew, 0)
            else: #human acts
                a_H = self.env.pi_H.get_action_from_state(state['state_H'], 0)
                rew = state['arms'][a_H].rvs() #sample reward from arm
                done = new_state['steps_taken'] >= self.env.wrapped_env.horizon
                new_state['state_H'] = self.env.pi_H.update_state(new_state['state_H'], 0, a_H, rew, 0)
        return (new_state, a_H, rew, done)

class BernoulliHumanBanditModel(Model):
    """ Black box simulator for humans playing bernoulli bandits, where reward is observed."""
    def __init__(self, env):
        assert (isinstance(env, DiscreteRewardWrapperEnv))
        self.env = env.wrapped_env
        self.bandit = env.wrapped_env.wrapped_env
    
    def generate_state_particles(self, n_particles):
        """
        Sample n state particles from the initial distribution. A particle is a dict containing a
            set of reward parameters for each arm and the human's internal state

        Args:
            n_particles (int): the number of particles to sample

        Returns:
            (list): the list of n particles. 
        """
        particles = []

        for i in range(n_particles):
            particle = {}
            arms = []
            theta = []
            for _, args in zip(range(self.bandit.n_arms), \
                               self.bandit.reward_args_generator):
                arms.append(self.bandit.reward_dist(**args))
                theta.append(args)
            particle['arms']= arms
            particle['theta'] = theta
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
        if isinstance(self.env, HumanCRLWrapper):
            if act == self.env.nA -1:
                a_H = self.env.pi_H.get_action_from_state(state['state_H'], 0)
                rew = new_state['arms'][a_H].rvs() - self.env.penalty #sample reward from arm
                done = new_state['steps_taken'] >= self.env.wrapped_env.horizon
                new_state['state_H'] = self.env.pi_H.update_state(new_state['state_H'], 0, a_H, rew, 0)
            else:
                a_H = self.env.nA - 1
                rew = new_state['arms'][act].rvs() #sample reward from arm
                done = new_state['steps_taken'] >= self.env.wrapped_env.horizon
                new_state['state_H'] = self.env.pi_H.update_state(new_state['state_H'], 0, act, rew, 0)
        elif isinstance(self.env, HumanIterativeWrapper):
            if new_state['steps_taken']%2 == 1: #robot acts
                a_H = self.env.nA
                rew = state['arms'][act].rvs() #sample reward from arm
                done = new_state['steps_taken'] >= self.env.wrapped_env.horizon
                new_state['state_H'] = self.env.pi_H.update_state(new_state['state_H'], 0, act, rew, 0)
            else: #human acts
                a_H = self.env.pi_H.get_action_from_state(state['state_H'], 0)
                rew = state['arms'][a_H].rvs() #sample reward from arm
                done = new_state['steps_taken'] >= self.env.wrapped_env.horizon
                new_state['state_H'] = self.env.pi_H.update_state(new_state['state_H'], 0, a_H, rew, 0)
        return (new_state, a_H*2+rew, rew, done)

class BanditPOMCPAgent(object):
    def __init__(self, env, pi_rollout=None, ucb_c=1, gamma=0.95, epsilon=0.01, n_particles=1000):
        """ Initializes this POMCP agent. """
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_particles = n_particles
        self.timesteps = 0
        self.ucb_c = ucb_c
        self.past_arm_counts = np.zeros(env.wrapped_env.n_arms, dtype=np.int32)

        self.model = HumanBanditModel(env)
        self.tree = BeliefTree(self.model, env.nA)
        self.tree.root.state_particles = self.model.generate_state_particles(self.n_particles)
        self.curr_node = self.tree.root
        
        self.last_act = None
        self.timesteps = 0
    
    def reset(self):
        self.timesteps = 0
        self.model = HumanBanditModel(self.env)
        
        self.tree.reset()
        self.tree.root.state_particles = self.model.generate_state_particles(self.n_particles)
        self.curr_node = self.tree.root
        
        self.past_arm_counts = np.zeros(self.env.wrapped_env.n_arms, dtype=np.int32)
        
        self.last_act = None
        
    def get_action(self, obs, max_simulations=5000, max_time=5):
        """
        Performs simulations for max_time seconds or until max_simulations simulations have been
            completed, then chooses the action with the highest value. 
        """
        if isinstance(obs, list):
            obs = self.env.observation_space.to_discrete(obs)
        
        if obs < self.env.wrapped_env.n_arms:
            self.past_arm_counts[obs]+=1
        
        if self.timesteps > 0:
            #Update belief state based on last observation
            next_node, _ = self.curr_node.create_or_get_child(self.last_act, obs)
            t0 = time.perf_counter()
            while len(next_node.state_particles) < self.n_particles:
                state = self.curr_node.sample_particle()
                new_state, sim_obs, rew, done = self.model.sample_transition(state, self.last_act)
                if obs == sim_obs:
                    next_node.state_particles.append(new_state)
                if time.perf_counter() - t0 > 10:
                    if len(next_node.state_particles) == 0:
                        print(self.last_act, obs)
                        print(rew)
                        print(sim_obs)
                        print(new_state)
                        raise RuntimeError('Monte Carlo Belief updates ended after '+ \
                                '{} particles due to timeout.'.format(len(next_node.state_particles)))
                    print('Monte Carlo Belief updates ended after ' + \
                                '{} particles due to timeout.'.format(len(next_node.state_particles)))
                    break
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
                    child_bn.state_particles.append(new_state)
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
        act = np.random.choice(np.where(self.past_arm_counts==np.max(self.past_arm_counts))[0])
        new_state, obs, rew, done = self.model.sample_transition(state, act)
        if done:
            return rew
        return rew + self.gamma * self.rollout(new_state, depth+1)
        
if __name__ == "__main__":
    bandit = BanditEnv(n_arms=4)
    pi_H = EpsGreedyBanditPolicy(bandit)
    env = HumanIterativeWrapper(bandit, pi_H)
    agent = BanditPOMCPAgent(env, n_particles=2000)
    rewards = []
    for i in range(5):
        observation = env.reset()
        print(env.wrapped_env.theta)
        for t in range(20):
            action = agent.get_action(observation, max_time=1)
            observation, reward, done, info = env.step(action)
            print("{}\t{}\t{}".format(action, observation, reward))
            if done:
                rewards.append(info['accumulated rewards'])
                state = agent.curr_node.sample_particle()
                print(state)
                print("Episode finished after {} timesteps".format(t+1))
                print("Cumulative reward:", info['accumulated rewards'])
                break
        agent.reset()
    print(np.mean(rewards), np.std(rewards))



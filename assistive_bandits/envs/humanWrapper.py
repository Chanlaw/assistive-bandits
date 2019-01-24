from __future__ import print_function, division

import numpy as np 
from scipy.stats import bernoulli, uniform, beta
import time 

import gym
from gym import Env
from gym.utils import seeding
from gym.envs.toy_text import discrete

from rllab.envs.base import EnvSpec
from sandbox.rocky.tf.spaces.box import Box #supercedes the gym space
from sandbox.rocky.tf.spaces.discrete import Discrete #supercedes the gym space
from assistive_bandits.spaces.multi_discrete import MultiDiscrete #supercedes the gym space


class HumanCRLWrapper(Env):
    """ 
    Takes a Discrete OpenAI gym environment and turns it into an iterative CRL game (from the robot's
    perspective). 

    Observations in this game consist of the current human observation and the last human action. 
    The robot's current observation is encoded as [obs_H, act_H]

    Actions in this game consist of the actions in the environment (encoded as 0 ... env.nA-1) 
    as well as a null op (encoded as env.nA). If the null op is taken, the human then acts. 

    Each time the human acts, both the human and the robot incur a penalty of fixed size, 
    encouraging the robot to act. Note that the robot does not observe reward during the game.
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, env, pi_H, penalty=0.1):
        """
        Initializes the environment. 
        
        Args:
            env (discrete.DiscreteEnv): a discrete gym environment. 
            penalty (float): the penalty applied to human moves. 
            pi_H (HumanPolicy): a human policy. 
        """
        if not isinstance(env, discrete.DiscreteEnv):
            raise ValueError("Environment provided is not discrete gym environment.")
            
        self.wrapped_env = env
        self.pi_H = pi_H
        self.penalty = penalty
        
        self.nA = env.nA + 1 #robot has same actions as human + null op
        self.nS = env.nS  #current human observation * last human action
        
        self._seed()
        self._reset()

    @property
    def observation_space(self):
        #note that min and max of multiDiscrete are inclusive. 
        return MultiDiscrete([[0, self.wrapped_env.nS-1],[0, self.wrapped_env.nA]]) 

    @property
    def action_space(self):
        return Discrete(self.nA)

    def get_param_values(self):
        ## TODO make this smarter. For now we're assuming that anytime we
        return None
    
    def _seed(self, seed=None):
        """ 
        Sets the random seed for both the wrapped environment, human policy, 
            and random choices made here.

        Args:
            seed (int): the seed to use. 

        returns:
            (list): list containing the random seed. 
        """
        self.np_random, seed = seeding.np_random(seed)
        self.wrapped_env.seed(seed)
        self.pi_H.seed(seed)
        return [seed]

    def reset(self):
        return self._reset()

    def _reset(self):
        """ 
        Resets the environment. 

        IT IS VERY IMPORTANT THAT wrapped_env.reset() is called before pi_H.reset(),
        since pi_H's reset may depend on the state of wrapped_env!!!

        Returns:
            [int, int]: the current state of the environment.
        """
        obs = self.wrapped_env.reset()

        self.pi_H.reset()
        
        self.done = False
        self.accumulated_rew = 0
        self.last_ob = obs
        
        return [obs, self.wrapped_env.nA]

    def _step(self, a):
        """ 
        Performs action a in the environment.

        Args:
            a (int): integer encoding the action to be taken. If a = env.nA, then this is 
                interpreted as the null op. If 0 < a < env.nA, then this is interpreted as the agent 
                acting in the environment with action a. Otherwise, an error is thrown.

        Returns: 
            (tuple): a tuple consisting of (obs, rew, done, info), where:
                obs ([int,int]): the current state of the environment. 
                rew (float): the reward obtained at this step. 
                            0 during the game, the cumulative reward over all the timesteps after. 
                done (bool): whether or not the wrapped environment is done. 
                info (dict): a dictionary of information for debugging.
        """
        if a < 0 or a > self.wrapped_env.nA:
            raise ValueError("{} is not a legal action".format(a))
        if (a == self.wrapped_env.nA): 
            a_H = self.pi_H.get_action(self.last_ob)
            obs_H, rew, done, info = self.wrapped_env.step(a_H)
            self.pi_H.learn(self.last_ob, a_H, rew, obs_H, done)
            self.last_ob = obs_H
            rew = rew - self.penalty
        else:
            a_H = self.wrapped_env.nA
            obs_H, rew, done, info = self.wrapped_env.step(a)
            self.pi_H.learn(self.last_ob, a, rew, obs_H, done)
            self.last_ob = obs_H
        self.done = done
        self.accumulated_rew += rew
        info['accumulated rewards'] = self.accumulated_rew
        return ([obs_H, a_H], 
                rew,
                self.done, 
                info)
            
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
    
class HumanIterativeWrapper(Env):
    """
    Human and robot take turns in the environment. 
    Human acts on even timesteps, robots act on odd timesteps.
    
    Observations in this game consist of the current human observation and the last human action. 
    The robot's current observation is encoded as [obs_H, act_H]. 
    
    Note that the robot does not observe the reward, though this can be remedied using the 
    rewardWrapper class. 
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, env, pi_H, **kwargs):
        """
        Initializes the environment. 
        Args:
            env (discrete.DiscreteEnv): a discrete gym environment. 
            pi_H (HumanPolicy): a human policy. 
        
        """
        if not isinstance(env, discrete.DiscreteEnv):
            raise ValueError("Environment provided is not discrete gym environment.")
        
        self.wrapped_env = env
        self.pi_H = pi_H
        
        self.nA = env.nA #robot has same actions as human
        self.nS = env.nS * env.nS * (env.nA + 1) 
                #current human observation * last human action
        
        self._seed()
        self._reset()

    @property
    def observation_space(self):
        #note that min and max of multiDiscrete are inclusive. 
        return MultiDiscrete([[0, self.wrapped_env.nS-1],[0, self.wrapped_env.nA]]) 

    @property
    def action_space(self):
        return Discrete(self.nA)

    def _seed(self, seed=None):
        """ 
        Sets the random seed for both the wrapped environment, human policy, 
            and random choices made here.

        Args:
            seed (int): the seed to use. 

        returns:
            (list): list containing the random seed. 
        """
        self.np_random, seed = seeding.np_random(seed)
        self.wrapped_env.seed(seed)
        self.pi_H.seed(seed)
        return [seed]
    
    def _reset(self):
        """ 
        Resets the environment. 

        IT IS VERY IMPORTANT THAT wrapped_env.reset() is called before pi_H.reset(),
        since pi_H's reset may depend on the state of wrapped_env!!!

        Returns:
            ([int, int]): the current state of the environment.
        """
        obs = self.wrapped_env.reset()
        
        self.pi_H.reset()
        
        self.timesteps = 0
        self.done = False
        self.accumulated_rew = 0
        self.last_ob = obs
        
        return [obs, self.wrapped_env.nA]
    
    def _step(self, a):
        """
        Take a step in the environment. If the timestep is odd, the action a is taken. 
            Otherwise, the human acts according to their own policy. 
        
        Args: 
            a (int): the action to take. 
        
        returns:
            (tuple): a tuple consisting of (obs, rew, done, info), where:
                obs ([int,int]): the current state of the environment. 
                rew (float): the reward obtained at this step. 
                            0 during the game, the cumulative reward over all the timesteps after. 
                done (bool): whether or not the wrapped environment is done. 
                info (dict): a dictionary of information for debugging.
        """
        self.timesteps += 1
        if self.timesteps %2 == 1:
            obs_H, rew, done, info = self.wrapped_env.step(a)
            a_H = self.wrapped_env.nA
            self.pi_H.learn(self.last_ob, a, rew, obs_H, done)
            self.last_ob = obs_H
        else:
            a_H = self.pi_H.get_action(self.last_ob)
            obs_H, rew, done, info = self.wrapped_env.step(a_H)
            self.pi_H.learn(self.last_ob, a_H, rew, obs_H, done)
            self.last_ob = obs_H
        self.done = done
        self.accumulated_rew += rew
        info['accumulated rewards'] = self.accumulated_rew
        return ([obs_H, a_H], 
                rew,
                self.done, 
                info)
    
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

class BetaBernoulliBanditCountWrapperEnv(Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, env, *args, max_timesteps=None):
        """
        Initialize the environment. 

        Args:
            env (Env): gym environment. Must have discrete observation and action spaces.
            max_timesteps (int): int indicating the max timesteps the environment will be run for.
        """
        assert(isinstance(env, BanditEnv))
        self.wrapped_env = env
        
        self.nA = env.action_space.n #actions are just the same actions as those in the environment. 
        self.state_dim = env.n_arms * 2
        
        self.counts = np.zeros(self.state_dim, dtype=np.int32) 
        
        if max_timesteps is not None:
            self.max_timesteps = max_timesteps
        else:
            max_timesteps = self.max_timesteps = env.horizon
        self.timesteps = 0
        self.Gittins = None
        self.action_space = Discrete(self.nA)
        obs_high = np.full(shape=self.counts.shape, fill_value=max_timesteps)
        self.observation_space = Box(np.zeros_like(self.counts), obs_high)
        self.dV_drhos = {}
        self._seed()
        
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
            (np.array): the current state of the environment. Of shape (self.state_dim*2 + 1,).
                        The first self.state_dim*2 entries represent 
        """
        self.counts = np.ones_like(self.counts)
        self.timesteps = 0
        obs = self.wrapped_env.reset()
        self.accumulated_rew = 0
        
        return self.counts.copy()
    
    def _step(self, a):
        """ 
        Performs action a in the environment.

        Args:
            a (int): integer encoding the action to be taken. 

        Returns: 
            (tuple): a tuple consisting of (obs, rew, done, info), where:
                obs (int): the current state of the environment. 
                rew (float): the reward obtained at this step. 
                            0 during the game, the cumulative reward over all the timesteps after. 
                done (bool): whether or not the wrapped environment is done. 
                info (dict): a dictionary of information for debugging. 
                        Currently only returns the human's Q-table.
        """
        obs, rew, done, info = self.wrapped_env.step(a)
        
        if rew > 0.5:
            self.counts[2*a] +=1
        else:
            self.counts[2*a+1]+=1
        self.timesteps += 1
        info['obs'] = obs
        self.accumulated_rew += rew
        info['accumulated rewards'] = self.accumulated_rew
        
        return (self.counts.copy(), rew, done, info)
        
        
    #Following things are used to get this to run with the rllab rnn-ppo code:
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
    
    def _compute_gittins_indices(self, discount=0.95):
        """
        Computes the Gittins Indices for the arms. 
        """
        print("Computing Gittins Index...")
        start_time = time.time()
        self.Gittins = gittins_index(n=100, grid=5000, discount=discount)
        print("Elapsed time in Gittins Index Calculation: ", time.time() - start_time)

    def _compute_arm_indices(self, state=None, discount=0.95):
        """
        Computes the Gittins indices associated with the arms 
        """
        if self.Gittins is None:
            self._compute_gittins_indices(discount=discount)
        if state is None:
            state = self.counts
        
        indices = np.array([self.Gittins[state[2*a]-1, state[2*a+1]-1] for a in range(self.nA)])
        return indices

    def _dV_drho(self, n_successes, n_failures, rho, discount=0.95):
        """
        Explicitly computes dV_i/drho by computing E[gamma^t(rho)]. We memoize the results for efficiency. 

        See Hadfield-Menell 2015.
        """
        if rho not in self.dV_drhos:
            self.dV_drhos[rho] = np.zeros(shape=(self.max_timesteps+1, self.max_timesteps+1))
        if rho > self.Gittins[n_successes-1, n_failures-1]/(1-discount) or \
                        n_successes + n_failures >= self.max_timesteps+1:
            return 1
        if self.dV_drhos[rho][n_successes, n_failures] > 0:
            return self.dV_drhos[rho][n_successes, n_failures]
        else:
            p = n_successes/(n_failures + n_successes)
            result = discount * (p * self._dV_drho(n_successes + 1, n_failures, rho, discount) \
                        + (1-p) * self._dV_drho(n_successes, n_failures + 1, rho, discount))
            self.dV_drhos[rho][n_successes, n_failures] = result
            return result


    def whittle_integral(self, state, rho=0, n_partitions=100, discount=0.95):
        """
        Approximately computes the whittle integral V(state, rho) by approximating
        the integrand with a step function. 
        """
        steps_taken = np.sum(state) - 2*self.nA
        arm_indices = self._compute_arm_indices(state=state)
        index = np.amax(arm_indices)*(self.max_timesteps - steps_taken + 1)
        if rho >= index:
            return rho
        partition = np.linspace(rho, index, n_partitions, endpoint=False)
        partition_size = (index-rho)/n_partitions
        whittle_integral = index
        for x in partition:
            integrand = np.prod([self._dV_drho(state[2*a], state[2*a+1], x, discount) \
                                                                 for a in range(self.nA)])
            whittle_integral -= partition_size * integrand
        return whittle_integral
    
    def _transitions(self, state, a, done):
        """ 
        Yields transitions from this state after taking the given action. 
        
        yields:
            (tuple): for each possible next state, this yields a tuple (prob, next_state, rew, done). 
        """
        p = state[2*a]/(state[2*a]+state[2*a+1])
        success_state = state.copy()
        success_state[2*a]+=1
        yield (p, tuple(success_state), 1, done)
        
        failure_state = state.copy()
        failure_state[2*a+1]+=1
        yield (1-p, tuple(failure_state), 0, done)
        
        
    def transitions(self, state):
        """ 
        Returns a dictionary which maps actions to lists of (prob, next_state, rew, done) tuples. 
        
        dict[a] = [(p1, s1, r1, done1), (p2, s2, r2, done1)...]
        """
        if isinstance(state, tuple):
            state = list(state)
        steps_taken = np.sum(state) - 2*self.nA
        if steps_taken >= self.max_timesteps:
            return {a: [] for a in range(self.nA)}
        done = (steps_taken + 1>= self.max_timesteps)
        transitions = {}
        for a in range(self.nA):
            transitions[a] = list(self._transitions(state, a, done))
        return transitions

class CRLIterativeBetaBernoulliBanditCountWrapperEnv(BetaBernoulliBanditCountWrapperEnv):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, env, pi_H, *args, max_timesteps=None):
        """
        Initialize the environment. 

        Args:
            env (Env): gym environment. Must have discrete observation and action spaces.
            max_timesteps (int): int indicating the max timesteps the environment will be run for.
        """
        self.pi_H = pi_H
        self.gittins_indices = None
        self.arm_indices = None
        super(CRLIterativeBetaBernoulliBanditCountWrapperEnv, self).__init__(env, max_timesteps=max_timesteps)
    
    def _step(self, a):
        """ 
        Performs action a in the environment.

        Args:
            a (int): integer encoding the action to be taken. 

        Returns: 
            (tuple): a tuple consisting of (obs, rew, done, info), where:
                obs (int): the current state of the environment. 
                rew (float): the reward obtained at this step. 
                            0 during the game, the cumulative reward over all the timesteps after. 
                done (bool): whether or not the wrapped environment is done. 
                info (dict): a dictionary of information for debugging. 
                        Currently only returns the human's Q-table.
        """
        self.timesteps += 1
        if self.timesteps %2 == 1:
            obs_H, rew, done, info = self.wrapped_env.step(a)
            a_H = self.wrapped_env.nA
            self.pi_H.learn(0, a, rew, 0)
            self.last_ob = obs_H
            if rew > 0.5:
                self.counts[2*a] +=1
            else:
                self.counts[2*a+1]+=1
        else:
            a_H = self.pi_H.get_action(self.last_ob)
            obs_H, rew, done, info = self.wrapped_env.step(a_H)
            self.pi_H.learn(0, a_H, rew, 0)
            self.last_ob = obs_H
            if rew > 0.5:
                self.counts[2*a_H] +=1
            else:
                self.counts[2*a_H+1]+=1
        self.done = done
        self.accumulated_rew += rew
        info['accumulated rewards'] = self.accumulated_rew
        
        
            
        if self.timesteps >= self.max_timesteps:
            done = True
        info['obs'] = obs_H
        
        
        return (self.counts.copy(), rew, done, info)
    
    def _transitions(self, state, a, done):
        """ 
        Yields transitions from this state after taking the given action. 
        
        yields:
            (tuple): for each possible next state, this yields a tuple (prob, next_state, rew, done). 
        """
        p_1 = state[2*a]/(state[2*a]+state[2*a+1])
        success_state = state.copy()
        success_state[2*a]+=1
        act_H_probs = self.pi_H.act_probs_from_counts(success_state, last_arm=a, last_rew=1)
        state_dict={} #used to calculate states reachable from multiple paths
        for a_H, p_H in enumerate(act_H_probs):
            p_2 = success_state[2*a_H]/(success_state[2*a_H] + success_state[2*a_H+1])
            new_state = success_state.copy()
            new_state[2*a_H] += 1
            yield (p_1*p_H*p_2, tuple(new_state), 2, done)
            
            new_state = success_state.copy()
            new_state[2*a_H+1] += 1
            state_dict[tuple(new_state)] = (p_1*p_H*(1-p_2), 1)
            
        failure_state = state.copy()
        failure_state[2*a+1]+=1
        act_H_probs = self.pi_H.act_probs_from_counts(failure_state, last_arm=a, last_rew=0)
        for a_H, p_H in enumerate(act_H_probs):
            p_2 = failure_state[2*a_H]/(failure_state[2*a_H]+failure_state[2*a_H+1])
            new_state = failure_state.copy()
            new_state[2*a_H] += 1
            tup_state = tuple(new_state)
            if tup_state in state_dict:
                q, _ = state_dict[tup_state]
                state_dict[tup_state] = ((1-p_1)*p_H*p_2 + q, 1)
            else:
                yield ((1-p_1)*p_H*p_2, tup_state, 1, done)
            
            new_state = failure_state.copy()
            new_state[2*a_H+1] += 1
            tup_state = tuple(new_state)
            yield ((1-p_1)*p_H*(1-p_2), tup_state, 0, done)
        
        for tup_state in state_dict:
            p, rew = state_dict[tup_state]
            yield (p, tup_state, rew, done)
    
    def _reset(self):
        """
        IT IS VERY IMPORTANT THAT wrapped_env.reset() is called before pi_H.reset(),
        since pi_H's reset may depend on the state of wrapped_env!!!
        """
        obs = super(CRLIterativeBetaBernoulliBanditCountWrapperEnv, self)._reset()
        self.pi_H.reset()
        return obs

    def transitions(self, state):
        """ 
        Returns a dictionary which maps actions to lists of (prob, next_state, rew, done) tuples. 
        
        dict[a] = [(p1, s1, r1, done1), (p2, s2, r2, done1)...]
        """
        if isinstance(state, tuple):
            state = list(state)
        steps_taken = np.sum(state) - 2*self.nA
        if steps_taken >= self.max_timesteps:
            return {a: [] for a in range(self.nA)}
        done = (steps_taken + 2>= self.max_timesteps)
        transitions = {}
        if steps_taken + 1 == self.max_timesteps:
            for a in range(self.nA):
                transitions[a] = \
                    list(super(CRLIterativeBetaBernoulliBanditCountWrapperEnv, self)._transitions(state, a, done))
        else:
            for a in range(self.nA):
                transitions[a] = list(self._transitions(state, a, done))
        return transitions
    
    #Following things are used to get this to run with the rllab rnn-ppo code:
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


class HumanTeleopWrapper(Env):
    """ 
    Observations in this game consist of the current human observation and the current human action. 
    The robot's current observation is encoded as [obs_H, act_H]

    Actions in this game consist of the actions in the environment (encoded as 0 ... env.nA-1),
    as well as a special action "defer" (encoded as env.nA). If the robot chooses "defer", the
    human action is taken immediately. In order to encourage the robot to act, a small penalty
    is applied if "defer" is chosen. 

    Note that unlike in the HumanCRLWrapper case, the robot gets to see what the human wants to do 
    before acting.

    Note that the robot does not observe reward during the game.
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, env, pi_H, penalty=0.):
        """
        Initializes the environment. 
        
        Args:
            env (discrete.DiscreteEnv): a discrete gym environment. 
            pi_H (HumanPolicy): a human policy. 
            penalty 
        """
        if not isinstance(env, discrete.DiscreteEnv):
            raise ValueError("Environment provided is not discrete gym environment.")
        
        self.penalty = penalty
        self.wrapped_env = env
        self.pi_H = pi_H
        
        self.nA = env.nA + 1 #robot has same actions as human + defer
        self.nS = env.nS  #current human observation * last human action
        
        self._seed()
        self._reset()

    @property
    def observation_space(self):
        #note that min and max of multiDiscrete are inclusive. 
        return MultiDiscrete([[0, self.wrapped_env.nS-1],[0, self.wrapped_env.nA]]) 

    @property
    def action_space(self):
        return Discrete(self.nA)

    def get_param_values(self):
        ## TODO make this smarter. For now we're assuming that anytime we
        return None
    
    def _seed(self, seed=None):
        """ 
        Sets the random seed for both the wrapped environment, human policy, 
            and random choices made here.

        Args:
            seed (int): the seed to use. 

        returns:
            (list): list containing the random seed. 
        """
        self.np_random, seed = seeding.np_random(seed)
        self.wrapped_env.seed(seed)
        self.pi_H.seed(seed)
        return [seed]

    def reset(self):
        return self._reset()

    def _reset(self):
        """ 
        Resets the environment. 

        IT IS VERY IMPORTANT THAT wrapped_env.reset() is called before pi_H.reset(),
        since pi_H's reset may depend on the state of wrapped_env!!!

        Returns:
            [int, int]: the current state of the environment.
        """
        obs = self.wrapped_env.reset()

        self.pi_H.reset()
        
        self.done = False
        self.accumulated_rew = 0
        self.last_ob = obs
        a_H = self.last_a_H = self.pi_H.get_action(obs)
        
        return [obs, a_H]

    def _step(self, a):
        """ 
        Performs action a in the environment.

        Args:
            a (int): integer encoding the action to be taken. If a < self.wrapped_env.nA, this is 
                    interpreted literally. Otherwise, we interpret this as deferring to the human's
                    last action.

        Returns: 
            (tuple): a tuple consisting of (obs, rew, done, info), where:
                obs ([int,int]): the current state of the environment. 
                rew (float): the reward obtained at this step. 
                            0 during the game, the cumulative reward over all the timesteps after. 
                done (bool): whether or not the wrapped environment is done. 
                info (dict): a dictionary of information for debugging.
        """
        if a < 0 or a > self.wrapped_env.nA:
            raise ValueError("{} is not a legal action".format(a))
        if a == self.wrapped_env.nA:
            a_H = self.last_a_H
            obs_H, rew, done, info = self.wrapped_env.step(a_H)
            self.pi_H.learn(self.last_ob, a_H, rew, obs_H, done)
            robot_rew = rew - self.penalty
        else:
            obs_H, rew, done, info = self.wrapped_env.step(a)
            self.pi_H.learn(self.last_ob, a, rew, obs_H, done)
            robot_rew = rew
        self.last_ob = obs_H
        self.done = done
        self.accumulated_rew += rew
        info['accumulated rewards'] = self.accumulated_rew
        a_H = self.last_a_H = self.pi_H.get_action(obs_H)
        return ([obs_H, a_H], 
                robot_rew,
                self.done, 
                info)
            
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
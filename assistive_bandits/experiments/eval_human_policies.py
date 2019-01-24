import os.path as osp
import pyprind
import numpy as np

import rllab.misc.logger as logger

from assistive_bandits.envs import BanditEnv
from assistive_bandits.envs.humanWrapper import HumanCRLWrapper, HumanIterativeWrapper
from assistive_bandits.envs.humanPolicy import HumanPolicy, EpsGreedyBanditPolicy, WSLSBanditPolicy, KGBetaBernoulliBanditPolicy,\
                                 TSBetaBernoulliBanditPolicy, GIBetaBernoulliBanditPolicy, EpsOptimalBanditPolicy,\
                                 UCLBetaBernoulliBanditPolicy
from assistive_bandits.envs.utils import uniform_bernoulli_iterator
from scipy.stats import bernoulli, uniform, beta

from assistive_bandits.envs.mdp import FrozenLakeEnv
from assistive_bandits.envs.humanPolicy import LinearQLearnerPolicy, LinearEpsGreedyPolicy, \
                                EpsOptimalMDPPolicy

human_policy_dict = {'eps-greedy': EpsGreedyBanditPolicy,
                     'wsls': WSLSBanditPolicy,
                     'knowledge-gradient': KGBetaBernoulliBanditPolicy, 
                     'thompson': TSBetaBernoulliBanditPolicy,
                     'gittins': GIBetaBernoulliBanditPolicy,
                     'eps-opt': EpsOptimalBanditPolicy,
                     'ucl':UCLBetaBernoulliBanditPolicy}

human_mdp_policies = {
    "eps-greedy": LinearEpsGreedyPolicy,
    "soft-max": LinearQLearnerPolicy,
    "eps-opt": EpsOptimalMDPPolicy
}

candidate_gammas = [0.7, 0.9, 0.97, 0.99, 0.997, 0.999]
def find_optimal_gamma(horizon=15, n_traj = 1000, map_name="5x5"):
    w_env = FrozenLakeEnv(map_name="9x9", horizon=horizon, theta_dist="hypercube")
    for gamma in candidate_gammas:
        test_pi_H = EpsOptimalMDPPolicy(w_env, discount=gamma)
        logger.log("-------------------")
        logger.log("Evaluating gamma={} for {} timesteps".format(gamma, horizon))
        logger.log("-------------------")
        test_env = HumanCRLWrapper(w_env, test_pi_H, 0)
        logger.log("Obtaining Samples...") 
            # Alas, the rllab samplers don't support hot swapping envs and batch sizes
            # TODO: write a new parallel sampler, instead of sampling manually
        rewards = []
        regrets = []
        for i in pyprind.prog_bar(range(n_traj)):
            observation = test_env.reset()
            for t in range(horizon):
                action = test_env.nA - 1
                observation, reward, done, info = test_env.step(action)
                if done:
                    rewards.append(info["accumulated rewards"])
                    regrets.append(info["accumulated regret"])
                    break
        #feel free to add more data 
        logger.log("NumTrajs {}".format(n_traj))
        logger.log("AverageReturn {}".format(np.mean(rewards)))
        logger.log("StdReturn {}".format(np.std(rewards)))
        logger.log("MaxReturn {}".format(np.max(rewards)))
        logger.log("MinReturn {}".format(np.min(rewards)))
        logger.log("AverageRegret {}".format(np.mean(regrets)))
        logger.log("MaxRegret {}".format(np.max(regrets)))
        logger.log("MinRegret {}".format(np.min(regrets)))

candidate_epsilons = [0.5, 0.3, 0.1, 0.05, 0.03, 0.01]

def find_optimal_epsilon(n_arms=4, horizon=15, n_traj=10000, log_dir=None):
    text_output_file = None if log_dir is None else osp.join(log_dir, "text")
    rag = uniform_bernoulli_iterator()
    bandit = BanditEnv(n_arms=n_arms, reward_dist=bernoulli, reward_args_generator=rag, horizon=horizon)
    if text_output_file is not None:
        logger.add_text_output(text_output_file)

    for epsilon in candidate_epsilons:
        # for i in range(10000):
        #     logger.log("Filler")

        logger.log("-------------------")
        logger.log("Evaluating epsilon={} for {} timesteps".format(epsilon, horizon))
        logger.log("-------------------")

        test_pi_H = EpsGreedyBanditPolicy(bandit, epsilon=epsilon)
        test_env = HumanCRLWrapper(bandit, test_pi_H, 0)
        logger.log("Obtaining Samples...") 
        # Alas, the rllab samplers don't support hot swapping envs and batch sizes
        # TODO: write a new parallel sampler, instead of sampling manually
        rewards = []
        for i in pyprind.prog_bar(range(n_traj)):
            observation = test_env.reset()
            for t in range(horizon):
                action = test_env.nA - 1
                observation, reward, done, info = test_env.step(action)
                if done:
                    rewards.append(info["accumulated rewards"])
                    break
        #feel free to add more data 
        logger.log("NumTrajs {}".format(n_traj))
        logger.log("AverageReturn {}".format(np.mean(rewards)))
        logger.log("StdReturn {}".format(np.std(rewards)))
        logger.log("MaxReturn {}".format(np.max(rewards)))
        logger.log("MinReturn {}".format(np.min(rewards)))

    if text_output_file is not None:
        logger.remove_text_output(text_output_file)

def eval_mdp_policies(horizon=15, n_traj=100000, log_dir=None):
    text_output_file = None if log_dir is None else osp.join(log_dir, "text")
    w_env = FrozenLakeEnv(horizon=horizon)
    if text_output_file is not None:
        logger.add_text_output(text_output_file)
    for human_policy in human_mdp_policies.values():
        logger.log("-------------------")
        logger.log("Evaluating {} for {} timesteps".format(human_policy.__name__, horizon))
        logger.log("-------------------")

        test_pi_H = human_policy(w_env)
        test_env = HumanCRLWrapper(w_env, test_pi_H)
        logger.log("Obtaining Samples...") 
        rewards = []
        for i in pyprind.prog_bar(range(n_traj)):
            observation = test_env.reset()
            for t in range(horizon):
                # _, action = observation
                # if action == test_env.nA:
                action = test_env.nA - 1
                observation, reward, done, info = test_env.step(action)
                if done:
                    rewards.append(info["accumulated rewards"])
                    break
        #feel free to add more data 
        logger.log("NumTrajs {}".format(n_traj))
        logger.log("AverageReturn {}".format(np.mean(rewards)))
        logger.log("StdReturn {}".format(np.std(rewards)))
        logger.log("MaxReturn {}".format(np.max(rewards)))
        logger.log("MinReturn {}".format(np.min(rewards)))

def eval_mab_policies(n_arms=4, horizon=15, n_traj=1000, log_dir=None, turntaking=False):
    text_output_file = None if log_dir is None else osp.join(log_dir, "text")
    rag = uniform_bernoulli_iterator()
    bandit = BanditEnv(n_arms=n_arms, reward_dist=bernoulli, reward_args_generator=rag, horizon=horizon)
    if text_output_file is not None:
        logger.add_text_output(text_output_file)

    for human_policy in [human_policy_dict['ucl']]: # human_policy_dict.values():
        # for i in range(10000):
        #     logger.log("Filler")

        logger.log("-------------------")
        logger.log("Evaluating {} for {} timesteps".format(human_policy.__name__, horizon))
        logger.log("-------------------")

        test_pi_H = human_policy(bandit)
        if turntaking:
            test_env = HumanIterativeWrapper(bandit, test_pi_H)
        else:
            test_env = HumanCRLWrapper(bandit, test_pi_H, 0)

        logger.log("Obtaining Samples...") 
        # Alas, the rllab samplers don't support hot swapping envs and batch sizes
        # TODO: write a new parallel sampler, instead of sampling manually
        rewards = []
        for i in pyprind.prog_bar(range(n_traj)):
            if turntaking:
                act_counts = [0 for i in range(bandit.nA)]
            observation = test_env.reset()
            action = test_env.nA - 1
            for t in range(horizon):
                observation, reward, done, info = test_env.step(action)
                if turntaking:
                    a_H = observation[1]
                    if a_H < bandit.nA:
                        act_counts[a_H] += 1
                    action = np.argmax(act_counts)
                if done:
                    rewards.append(info["accumulated rewards"])
                    break
        #feel free to add more data 
        logger.log("NumTrajs {}".format(n_traj))
        logger.log("AverageReturn {}".format(np.mean(rewards)))
        logger.log("StdReturn {}".format(np.std(rewards)))
        logger.log("MaxReturn {}".format(np.max(rewards)))
        logger.log("MinReturn {}".format(np.min(rewards)))

    if text_output_file is not None:
        logger.remove_text_output(text_output_file)
if __name__ == "__main__":
    eval_mab_policies(horizon=50, n_traj=10000, turntaking=True)

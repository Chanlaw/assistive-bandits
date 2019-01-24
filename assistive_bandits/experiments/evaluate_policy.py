import argparse
import os
import os.path as osp

import joblib
import tensorflow as tf
import numpy as np

from rllab import config
from rllab.sampler import parallel_sampler
from rllab.misc.instrument import VariantGenerator, variant, run_experiment_lite, stub
import rllab.misc.logger as logger
from sandbox.rocky.tf.samplers.vectorized_sampler import VectorizedSampler

from assistive_bandits.envs import BanditEnv
from assistive_bandits.envs.humanWrapper import HumanCRLWrapper, HumanIterativeWrapper, HumanTeleopWrapper
from assistive_bandits.envs.humanPolicy import HumanPolicy, EpsGreedyBanditPolicy, WSLSBanditPolicy, \
                                KGBetaBernoulliBanditPolicy, TSBetaBernoulliBanditPolicy, GIBetaBernoulliBanditPolicy, \
                                EpsOptimalBanditPolicy, UCLBetaBernoulliBanditPolicy
from assistive_bandits.experiments.pposgd_clip_ratio import PPOSGD


from assistive_bandits.envs.utils import uniform_bernoulli_iterator
from assistive_bandits.envs.humanPolicy.infoTheoryUtils import _frequency_agreement, _mutual_info_seqs
from scipy.stats import bernoulli, uniform, beta

human_policy_dict = {'eps-greedy': EpsGreedyBanditPolicy,
                     'wsls': WSLSBanditPolicy,
                     # 'kg': KGBetaBernoulliBanditPolicy, 
                     'thompson': TSBetaBernoulliBanditPolicy,
                     'gittins': GIBetaBernoulliBanditPolicy,
                     'eps-opt': EpsOptimalBanditPolicy,
                     'ucl': UCLBetaBernoulliBanditPolicy}

human_wrapper_dict = {"preemptive": HumanCRLWrapper, "turn-taking": HumanIterativeWrapper,
                      "teleop": HumanTeleopWrapper}

local_test=True


num_eval_traj = 1000 if local_test else 100000

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('-T', '--max_path_length', type=int, default=10)
    parser.add_argument('-nt', '--num_eval_traj', type=int, default=1000,
    					help="number of trajectories to evaluate for")
    parser.add_argument('--eval_against', 
    					choices=['eps-greedy', 'wsls', 'thompson', 'gittins', 'ucl', 'eps-opt', 'all', 'self'],
						default='all')
    parser.add_argument('-l', '--log_dir', type=str, default="./data")
    args = parser.parse_args()
    log_dir = args.log_dir

    num_eval_traj = args.num_eval_traj 

    text_output_file = None if log_dir is None else osp.join(log_dir, "text")
    info_theory_tabular_output = None if log_dir is None else osp.join(log_dir, "info_table.csv")

    if text_output_file is not None:
        logger.add_text_output(text_output_file)
        logger.add_tabular_output(info_theory_tabular_output)

    with tf.Session() as sess:
        data = joblib.load(args.file)
        policy = data['policy']
        env = data['env']
        algo = PPOSGD(env=env,
        			baseline=data['baseline'],
        			policy=policy)
        bandit=env.wrapped_env
        logger.log("Loaded policy trained against {} for {} timesteps".format(env.pi_H.__class__.__name__,
        																	 bandit.horizon))
       	max_timesteps = min(bandit.horizon, args.max_path_length)

       	if args.eval_against == 'all':
       		itr = human_policy_dict.items()
        elif args.eval_against == 'self':
            itr = [(args.eva_against, None)]
       	else:
       		itr = [(args.eval_against, human_policy_dict[args.eval_against])]


       	for human_policy_name, human_policy in itr:

            logger.log("-------------------")
            logger.log("Evaluating against {}".format(human_policy.__name__))
            logger.log("-------------------")

            logger.log("Obtaining Samples...") 

            if human_policy_name=="self":
                test_env = env
            else:
                test_pi_H = human_policy(bandit)
                test_env = HumanTeleopWrapper(bandit, test_pi_H, penalty=0.)
            eval_sampler = VectorizedSampler(algo, n_envs=100)
            algo.max_path_length = max_timesteps
            algo.batch_size = num_eval_traj*max_timesteps
            algo.env = test_env
            logger.log("algo.env.pi_H has class: {}".format(algo.env.pi_H.__class__))
            eval_sampler.start_worker()
            paths = eval_sampler.obtain_samples(-1)
            eval_sampler.shutdown_worker()
            rewards = []

            H_act_seqs = []
            R_act_seqs = []
            best_arms = []
            optimal_a_seqs = []
            for p in paths:
                a_Rs = env.action_space.unflatten_n(p['actions'])
                obs_R = env.observation_space.unflatten_n(p['observations'])
                best_arm = np.argmax(p['env_infos']['arm_means'][0])

                H_act_seqs.append(obs_R[:,1])
                R_act_seqs.append(a_Rs)
                best_arms.append(best_arm)
                optimal_a_seqs.append([best_arm for _ in range(max_timesteps)])

                rewards.append(np.sum(p['rewards']))

            #feel free to add more data 
            logger.log("NumTrajs {}".format(num_eval_traj))
            logger.log("AverageReturn {}".format(np.mean(rewards)))
            logger.log("StdReturn {}".format(np.std(rewards)))
            logger.log("MaxReturn {}".format(np.max(rewards)))
            logger.log("MinReturn {}".format(np.min(rewards)))

            optimal_a_H_freqs = _frequency_agreement(H_act_seqs, optimal_a_seqs)
            optimal_a_R_freqs = _frequency_agreement(R_act_seqs, optimal_a_seqs)

            for t in range(max_timesteps):
                logger.record_tabular("PolicyExecTime", 0)
                logger.record_tabular("EnvExecTime", 0)
                logger.record_tabular("ProcessExecTime", 0)

                logger.record_tabular("Tested Against", human_policy_name)
                logger.record_tabular("t", t)
                logger.record_tabular("a_H_agreement", optimal_a_H_freqs[t])
                logger.record_tabular("a_R_agreement", optimal_a_R_freqs[t])

                H_act_seqs_truncated = [a_Hs[0:t] for a_Hs in H_act_seqs]
                R_act_seqs_truncated = [a_Rs[0:t] for a_Rs in R_act_seqs]
                h_mutual_info = _mutual_info_seqs(H_act_seqs_truncated, best_arms, bandit.nA+1)
                r_mutual_info = _mutual_info_seqs(R_act_seqs_truncated, best_arms, bandit.nA+1)
                logger.record_tabular("h_mutual_info", h_mutual_info)
                logger.record_tabular("r_mutual_info", r_mutual_info)
                logger.record_tabular("a_H_opt_freq", optimal_a_H_freqs[t])
                logger.record_tabular("a_R_opt_freq", optimal_a_R_freqs[t])
                logger.dump_tabular()

    if text_output_file is not None:
        logger.remove_text_output(text_output_file)
        logger.remove_tabular_output(info_theory_tabular_output)


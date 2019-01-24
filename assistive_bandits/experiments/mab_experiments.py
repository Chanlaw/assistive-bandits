import os
import os.path as osp
import datetime
import dateutil.tz
import numpy as np 
import pyprind

from rllab import config
from rllab.sampler import parallel_sampler
from rllab.misc.instrument import VariantGenerator, variant, run_experiment_lite, stub
import rllab.misc.logger as logger
from sandbox.rocky.tf.envs.vec_env_executor import VecEnvExecutor
from sandbox.rocky.tf.samplers.vectorized_sampler import VectorizedSampler

from assistive_bandits.envs import BanditEnv
from assistive_bandits.envs.humanWrapper import HumanCRLWrapper, HumanIterativeWrapper, HumanTeleopWrapper
from assistive_bandits.envs.humanPolicy import HumanPolicy, EpsGreedyBanditPolicy, WSLSBanditPolicy, \
                                KGBetaBernoulliBanditPolicy, TSBetaBernoulliBanditPolicy, GIBetaBernoulliBanditPolicy, \
                                EpsOptimalBanditPolicy, AnnealedTSBBBPolicy,\
                                UCLBetaBernoulliBanditPolicy
from assistive_bandits.envs.utils import uniform_bernoulli_iterator
from assistive_bandits.envs.humanPolicy.infoTheoryUtils import _frequency_agreement, _mutual_info_seqs
from scipy.stats import bernoulli, uniform, beta


human_policy_dict = {'eps-greedy': EpsGreedyBanditPolicy,
                     'wsls': WSLSBanditPolicy,
                     'ucl': UCLBetaBernoulliBanditPolicy,
                     # 'kg': KGBetaBernoulliBanditPolicy, 
                     'thompson': TSBetaBernoulliBanditPolicy,
                     'gittins': GIBetaBernoulliBanditPolicy,
                     'eps-opt': EpsOptimalBanditPolicy,
                     'annealedTS': AnnealedTSBBBPolicy}

human_wrapper_dict = {"preemptive": HumanCRLWrapper, "turn-taking": HumanIterativeWrapper,
                      "teleop": HumanTeleopWrapper}

local_test=False
force_remote=False

if force_remote and not local_test:
    from feedback_imitation.experiment_utils import run_remotely


class mab_VG(VariantGenerator):
    @variant
    def seed(self):
        #random seeds for statistical reasons
        # return [14, 28, 32, 37, 17]
        return [16]#, 28, 32]#, 37, 17]
    
    @variant
    def human_wrapper(self):
        return ['teleop']
        #return human_wrapper_dict.keys()
        # return ['preemptive']

    @variant
    def batch_size(self):
        if local_test:
            return [10000]
        return [250000]

    @variant
    def clip_lr(self):
        return [0.2]

    @variant
    def use_kl_penalty(self):
        return [False]

    @variant
    def nonlinearity(self):
        return ["relu"]

    @variant
    def n_arms(self):
        return [4]

    @variant
    def mean_kl(self):
        return [0.01]

    @variant
    def layer_normalization(self):
        return [False]

    @variant
    def n_episodes(self):
        # if local_test:
        #     return [15]
        return [50]

    @variant
    def weight_normalization(self):
        return [True]

    @variant
    def min_epochs(self):
        return [5]

    @variant
    def opt_batch_size(self):
        return [128]

    @variant
    def opt_n_steps(self):
        return [None]

    @variant
    def batch_normalization(self):
        return [False]

    @variant
    def entropy_bonus_coeff(self):
        return [0]

    @variant
    def discount(self):
        # Discount factor. For variance reduction. We don't actually need a discount rate. 
        return [1.0]

    @variant
    def gae_lambda(self):
        # Discount factor for future Bellman residuals. Used for Generalized Advantage Estimation.
        return [0.3]

    @variant
    def hidden_dim(self):
        return [256]

    @variant
    def human_policy(self):
        # return human_policy_dict.keys()
        return ["eps-opt"]

    @variant
    def intervention_penalty(self):
        return [0.01]
        #return [0.]

    @variant
    def num_eval_traj(self):
        if local_test:
            return [1000]
        return [1000000]

    @variant
    def num_display_traj(self):
        if local_test:
            return [2]
        return [10]

def run_task(v, num_cpu=8, log_dir="./data", ename=None,  **kwargs):
    from scipy.stats import bernoulli, uniform, beta

    import tensorflow as tf
    from assistive_bandits.experiments.l2_rnn_baseline import L2RNNBaseline
    from assistive_bandits.experiments.tbptt_optimizer import TBPTTOptimizer
    from sandbox.rocky.tf.policies.categorical_gru_policy import CategoricalGRUPolicy
    from assistive_bandits.experiments.pposgd_clip_ratio import PPOSGD

    if not local_test and force_remote:
        import rl_algs.logger as rl_algs_logger
        log_dir = rl_algs_logger.get_dir()

    if log_dir is not None:
        log_dir = osp.join(log_dir, str(v["n_episodes"]))
        log_dir = osp.join(log_dir, v["human_policy"])

    text_output_file = None if log_dir is None else osp.join(log_dir, "text")
    tabular_output_file = None if log_dir is None else osp.join(log_dir, "train_table.csv")
    info_theory_tabular_output = None if log_dir is None else osp.join(log_dir, "info_table.csv")
    rag = uniform_bernoulli_iterator()
    bandit = BanditEnv(n_arms=v["n_arms"], reward_dist=bernoulli, reward_args_generator=rag, horizon=v["n_episodes"])
    pi_H = human_policy_dict[v["human_policy"]](bandit)

    h_wrapper = human_wrapper_dict[v["human_wrapper"]]
    
    env = h_wrapper(bandit, pi_H, penalty=v["intervention_penalty"])

    if text_output_file is not None:
        logger.add_text_output(text_output_file)
        logger.add_tabular_output(tabular_output_file)

    logger.log("Training against {}".format(v["human_policy"]))
    logger.log("Setting seed to {}".format(v["seed"]))
    env.seed(v["seed"])

    baseline = L2RNNBaseline(
            name="vf",
            env_spec=env.spec,
            log_loss_before=False,
            log_loss_after=False,
            hidden_nonlinearity=getattr(tf.nn, v["nonlinearity"]),
            weight_normalization=v["weight_normalization"],
            layer_normalization=v["layer_normalization"],
            state_include_action=False,
            hidden_dim=v["hidden_dim"],
            optimizer=TBPTTOptimizer(
                batch_size=v["opt_batch_size"],
                n_steps=v["opt_n_steps"],
                n_epochs=v["min_epochs"],
            ),
            batch_size=v["opt_batch_size"],
            n_steps=v["opt_n_steps"],
        )
    policy = CategoricalGRUPolicy(
            env_spec=env.spec,
            hidden_nonlinearity=getattr(tf.nn, v["nonlinearity"]),
            hidden_dim=v["hidden_dim"],
            state_include_action=True,
            name="policy"
        )

    n_itr = 3 if local_test else 100
    # logger.log('sampler_args {}'.format(dict(n_envs=max(1, min(int(np.ceil(v["batch_size"] / v["n_episodes"])), 100)))))

    # parallel_sampler.initialize(6)
    # parallel_sampler.set_seed(v["seed"])

    algo = PPOSGD(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=v["batch_size"],
            max_path_length=v["n_episodes"],
            # 43.65 env time
            sampler_args=dict(n_envs=max(1, min(int(np.ceil(v["batch_size"] / v["n_episodes"])), 100))),
            # 100 threads -> 1:36 to sample 187.275
            # 6 threads -> 1:31
            # force_batch_sampler=True,            
            n_itr=n_itr,
            step_size=v["mean_kl"],
            clip_lr=v["clip_lr"],
            log_loss_kl_before=False,
            log_loss_kl_after=False,
            use_kl_penalty=v["use_kl_penalty"],
            min_n_epochs=v["min_epochs"],
            entropy_bonus_coeff=v["entropy_bonus_coeff"],
            optimizer=TBPTTOptimizer(
                batch_size=v["opt_batch_size"],
                n_steps=v["opt_n_steps"],
                n_epochs=v["min_epochs"],
            ),
            discount=v["discount"],
            gae_lambda=v["gae_lambda"],
            use_line_search=True
            # scope=ename
        )

    sess = tf.Session()
    with sess.as_default():
        algo.train(sess)

        if text_output_file is not None:
            logger.remove_tabular_output(tabular_output_file)
            logger.add_tabular_output(info_theory_tabular_output)

        # Now gather statistics for t-tests and such!
        for human_policy_name, human_policy in human_policy_dict.items():

            logger.log("-------------------")
            logger.log("Evaluating against {}".format(human_policy.__name__))
            logger.log("-------------------")

            logger.log("Obtaining Samples...") 
            test_pi_H = human_policy(bandit)
            test_env = h_wrapper(bandit, test_pi_H, penalty=0.)
            eval_sampler = VectorizedSampler(algo, n_envs=100)
            algo.batch_size = v["num_eval_traj"] * v["n_episodes"]
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
                optimal_a_seqs.append([best_arm for _ in range(v["n_episodes"])])

                rewards.append(np.sum(p['rewards']))

            #feel free to add more data 
            logger.log("NumTrajs {}".format(v["num_eval_traj"]))
            logger.log("AverageReturn {}".format(np.mean(rewards)))
            logger.log("StdReturn {}".format(np.std(rewards)))
            logger.log("MaxReturn {}".format(np.max(rewards)))
            logger.log("MinReturn {}".format(np.min(rewards)))

            optimal_a_H_freqs = _frequency_agreement(H_act_seqs, optimal_a_seqs)
            optimal_a_R_freqs = _frequency_agreement(R_act_seqs, optimal_a_seqs)

            for t in range(v["n_episodes"]):
                logger.record_tabular("PolicyExecTime", 0)
                logger.record_tabular("EnvExecTime", 0)
                logger.record_tabular("ProcessExecTime", 0)
                logger.record_tabular("Tested Against", human_policy_name)
                logger.record_tabular("t", t)
                logger.record_tabular("a_H_agreement", optimal_a_H_freqs[t])
                logger.record_tabular("a_R_agreement", optimal_a_R_freqs[t])

                H_act_seqs_truncated = [a_Hs[0:t] for a_Hs in H_act_seqs]
                R_act_seqs_truncated = [a_Rs[0:t] for a_Rs in R_act_seqs]
                h_mutual_info = _mutual_info_seqs(H_act_seqs_truncated, best_arms, v["n_arms"]+1)
                r_mutual_info = _mutual_info_seqs(R_act_seqs_truncated, best_arms, v["n_arms"]+1)
                logger.record_tabular("h_mutual_info", h_mutual_info)
                logger.record_tabular("r_mutual_info", r_mutual_info)
                logger.record_tabular("a_H_opt_freq", optimal_a_H_freqs[t])
                logger.record_tabular("a_R_opt_freq", optimal_a_R_freqs[t])
                logger.dump_tabular()


            test_env = h_wrapper(bandit, human_policy(bandit), penalty=0.)

            logger.log("Printing Example Trajectories")
            for i in range(v["num_display_traj"]):
                observation = test_env.reset()
                policy.reset()
                logger.log("-- Trajectory {} of {}".format(i+1, v["num_display_traj"]))
                logger.log("t \t obs \t act \t reward \t act_probs")
                for t in range(v["n_episodes"]):
                    action, act_info = policy.get_action(observation)
                    new_obs, reward, done, info = test_env.step(action)
                    logger.log("{} \t {} \t {} \t {} \t {}".format(t,
                                 observation, action, reward, act_info['prob']))
                    observation = new_obs
                    if done:
                        logger.log("Total reward: {}".format(info["accumulated rewards"]))
                        break

    if text_output_file is not None:
        logger.remove_text_output(text_output_file)
        logger.remove_tabular_output(info_theory_tabular_output)
        

def generate_experiments(n_cpus=16, n_gpus=1):
    """
    Returns a list of (name, task, task_arg) tuples. The tasks can be run by calling
    task(**task_arg).
    """
    vg = mab_VG()
    variants = vg.variants()
    logger.log("Generating {} experiments".format(len(variants)))
    experiments = []
    for v in variants:
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        # timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
        exp_name = "kg-t-{}_{}".format(v["seed"], v["n_episodes"])
        experiments.append((exp_name, run_task, ({'v':v, 'num_cpu':n_cpus, 'num_gpu': n_gpus, 'ename': exp_name})))

    return experiments

if __name__ == "__main__":
    config.AWS_S3_PATH = ""
    config.USE_TF = True
    if local_test:
        vg = mab_VG()
        variants = vg.variants()
        for v in variants:
            print(v)
            run_experiment_lite(run_task, 
                                n_parallel=4, 
                                exp_prefix=v["human_wrapper"], 
                                exp_name=v["human_policy"], 
                                variant=v)
    elif not force_remote:
        vg = mab_VG()
        variants = vg.variants()
        for v in variants:
            run_experiment_lite(run_task, 
                                n_parallel=8, 
                                exp_prefix=v["human_wrapper"], 
                                exp_name=v["human_policy"], 
                                variant=v, use_gpu=True)
    else:
        mode=os.environ.get('mode', 'local')
        experiments = generate_experiments(n_cpus=16, n_gpus=4)
        print('got experiments')
        for ename, f, kwargs in experiments:
            print(kwargs)
            run_remotely(f, 'teleop_gittins_rerun', eval_name=ename, mode=mode, **kwargs)
            if local_test:
                break

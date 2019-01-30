# Assistive Bandits
Environment and solver code for [The Assistive Multi-Armed Bandit (2019)](https://arxiv.org/abs/1901.08654).

## Introduction to our framework
Traditionally, techniques for preference learning assume that human behavior can be modelled as (noisily-)optimal with respect to a set of fixed preferences. However, our revealed preferences change all the time - especially in cases where we might not be fully aware of our preferences! For example, when we're organizing our desks, we might experiment with different configurations, to see what works.

Now imagine a personal robot is trying to help you organize that same desk. If the robot believes you are optimal, it will infer the wrong preferences. Instead, if the robot accounts for the fact that you are learning about your preferences, it has a better shot at understanding what you want. Even more crucially, the robot can expose you to new configurations -- ones that might improve your posture, something you hadn't considered before and were not going to explore if left to your own devices. 

<figure>
<img src="./figures/ab_teaser_v4.png" width="600">
<figcaption>The setup of the assistive bandit framework. In each round, the human observes reward and tells the robot which arm they would like it to pull. The robot observes these requests, attempts to infer the reward values, and selects an arm to pull. </figcaption>
</figure>

The assistive bandit framework seeks to formalize the task of helping an agent that is still learning about their preferences. It extends the [Multi-Armed Bandit](https://en.wikipedia.org/wiki/Multi-armed_bandit) model of reinforcement learning, and is heavily inspired by the [Cooperative Inverse Reinforcement Learning](https://arxiv.org/abs/1606.03137) framework. In each round, the human selects an action, referred to in the bandit setting as an arm.  However, the robot intercepts their intended action and chooses a (potentially different) arm to pull. The human then observes the pulled arm and corresponding reward, and the process repeats.

Despite appearing simple, we believe the framework captures interesting issues of information assymmetry and preference uncertainty that lie at the heart of assisting a learning agent. (See [our paper](https://arxiv.org/abs/1901.08654) for more details!)

### "Human" Policies
The super class for all Bandit policies is [`HumanPolicy`](https://github.com/Chanlaw/assistive-bandits/blob/master/assistive_bandits/envs/humanPolicy/humanPolicy.py). Currently, the following Bandit policies have been implemented:
- The [*random* policy](https://github.com/Chanlaw/assistive-bandits/blob/master/assistive_bandits/envs/humanPolicy/humanPolicy.py), which pulls arms entirely at random. 
- The [*epsilon-greedy* policy](https://github.com/Chanlaw/assistive-bandits/blob/master/assistive_bandits/envs/humanPolicy/humanPolicy.py). With probability epsilon, sample a random arm. Otherwise, pull the arm with highest empirical mean.
- The [*win-stay-lose-shift* policy](https://github.com/Chanlaw/assistive-bandits/blob/master/assistive_bandits/envs/humanPolicy/humanPolicy.py). If the reward gotten from an arm is greater than the mean (over all arms), keep pulling the arm. Otherwise, pick another arm to pull at random. 
- The [*Thompson sampling* policy](https://github.com/Chanlaw/assistive-bandits/blob/master/assistive_bandits/envs/humanPolicy/thompsonSamplingHumanPolicy.py). Maintain a posterior over each arm's parameters, and pull arms with probability equal to the chance the arm is optimal. This is implemented by sampling a particle from the posterior over arm means, then picking the arm with the highest mean in that particle. In addition, we also implement an annealed variant where many particles are sampled, and the arm with the highest mean amongst sampled particles is picked. Currently, the implementation assumes the bandit is Beta-Bernoulli - that is, the arm rewards are sampled from independent Bernoulli distributions, with parameters distributed according to a Beta distribution. 
- The [*knowledge gradient* policy](https://github.com/Chanlaw/assistive-bandits/blob/master/assistive_bandits/envs/humanPolicy/knowledgeGradientHumanPolicy.py). The knowledge gradient policy assumes (falsely) that the current timestep is the last opportunity to learn, and that the policy will continue to act later without any learning. It therefore picks an arm that maximizes the expected reward of the current timestep, plus the discounted sum of rewards of always pulling the best arm after this timestep. Currently, the implementation assumes the bandit is Beta-Bernoulli.
- The [*Gittins index* policy](https://github.com/Chanlaw/assistive-bandits/blob/master/assistive_bandits/envs/humanPolicy/gittinsIndexPolicy.py). This is the Bayes-optimal solution to a discounted, infinite horizon bandit. We use the approximation method of [Chakravorty and Mahajan](https://www.jhelumch.com/wp-content/uploads/2018/01/Gittins_indices_survey2013.pdf). Currently, the implementation assumes the bandit is Beta-Bernoulli.
- The [*upper confidence bound* (UCB) policy](https://github.com/Chanlaw/assistive-bandits/blob/master/assistive_bandits/envs/humanPolicy/UCBPolicy.py), which maintains upper confidence bounds on the mean of each arm, picks the arm with the highest upper confidence bound. 
- The [*Bayesian upper confidence bound* (Bayes-UCB) policy](https://github.com/Chanlaw/assistive-bandits/blob/master/assistive_bandits/envs/humanPolicy/UCBPolicy.py), which maintains posteriors over the mean of each arm, and picks the arm with the highest (1-1/t)-quantile. Currently, the implementation assumes the bandit is Beta-Bernoulli.
- The [*upper credible limit* policy](https://github.com/Chanlaw/assistive-bandits/blob/master/assistive_bandits/envs/humanPolicy/UCLPolicy.py), which is very similar to Bayes-UCB, but with softmax arm selection noise. Currently, the implementation assumes the bandit is Beta-Bernoulli.
- The [*epsilon-optimal* policy](https://github.com/Chanlaw/assistive-bandits/blob/master/assistive_bandits/envs/humanPolicy/epsOptimalPolicy.py), which has full knowledge of the arm means. It pulls a random arm with probability epsilon. Otherwise, it pulls the arm with the highest mean. 

### Alternative interaction modes
In addition to the Assistive bandit setup (`HumanTeleopWrapper`), we also implement the following two interaction modes in [`HumanWrapper.py`](https://github.com/Chanlaw/assistive-bandits/blob/master/assistive_bandits/envs/humanWrapper.py):
- Preemptive: the robot must choose which arm to pull *before* seeing the human's desired action, and only sees the human action if the robot did not act. This is implemented by `HumanCRLWrapper`
- Turn-taking: the robot pulls arms during even timesteps, while the human pulls during odd timesteps. This is implemented by `HumanIterativeWrapper`. 

<!-- ### Example policy

<img src="figures/wsls_near_opt.png" width="600"> -->

## Usage

#### Installation Requirements
```
gym (0.9.x)
numpy
scipy
rllab
tensorflow 1.2+
sklearn
pyprind
```
### (Recommended) Installing using install script
We recommend that you set up an isolated environment to install and run this code in, as it depends on an older version of `gym`.

First, setup your virtual environment. install `tensorflow` or `tensorflow-gpu`. Then, run:
```
chmod u+x install.sh
./install.sh
```

### Reproducing MAB results
To reproduce our results from The Assistive MAB paper, run
```
python assistive-bandits/experiments/mab-experiments.py 
```
By default, our code will sequentially train an agent against every human policy included in the paper, and then test this policy against every human policy. This takes a significant amount of time (about 2-3 hours per policy per seed on an AWS p3.2xlarge). 

#### Evaluating trained policies
To evaluate a trained assistive policy, run:
`python assistive-bandits/experiments/evaluate_policy.py <path to policy file>`

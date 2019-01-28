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

### Alternative interaction modes

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

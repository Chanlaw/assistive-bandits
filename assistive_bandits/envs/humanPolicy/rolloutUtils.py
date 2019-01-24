from __future__ import print_function, division

from pyprind import prog_bar
import numpy as np

def gen_human_bandit_trajs(pi_H, bandit_env, max_timesteps=None, n_trajs=10):
    """
    Generates n_traj trajectories of pi_H in bandit_env, of up to length max_timesteps. 
    No fancy tricks, just rolling them out one at a time. 
    
    Args:
        pi_H (HumanPolicy): a human policy to evaluate the best arms against. 
        bandit_env (BanditEnv): the bandit environment to run this human in. We assume 
                                that bandit_env has the same characteristics as pi_H.env.
        max_timesteps (int): the maximum amount of timesteps to run pi_H in the environment. 
        n_trajs (int): the number of trajectories to generate. 
    
    Returns:
        (dict): a list of dicts with keys "length", "acts", "obs", "rews", "best_arm", "total_rew". 
                Each dictionary contains the information associated with one trajectories. 
                - length (int): the length of this trajectories
                - acts (list-like): a list of all the actions taken by the human
                - obs (list-like): a list of observations made by human
                - rews (list-like): the rewards received by the human
                - best_arm (int): the index of the best arm of the bandit
                - total_rew (float): the total reward associated with this trajectory
                
    """
    if max_timesteps==None:
        max_timesteps = bandit_env.horizon
        
    trajs = []
    for i in prog_bar(range(n_trajs)):
        ob = bandit_env.reset()
        pi_H.reset()
        
        best_arm = np.argmax([bandit_env.arms[i].mean() for i in range(bandit_env.n_arms)])
        
        obs = []
        acts = []
        rews = []
        
        ob = 0 
        
        for t in range(max_timesteps):
            obs.append(ob)
            
            act = pi_H.get_action(ob)
            acts.append(act)
            
            ob, rew, done, info = bandit_env.step(act)
            rews.append(rew)
            
            pi_H.learn(obs[-1], act, rew, ob, done)
            if done:
                break
                
        traj = {"length": t, "acts":acts, "obs":obs, "rewards":rews, 
                "best_arm":best_arm, "total_rew": sum(rews)}
        trajs.append(traj)
        
    return trajs

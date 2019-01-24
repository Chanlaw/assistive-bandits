from __future__ import print_function, division

from pyprind import prog_bar
import numpy as np
from sklearn.metrics import mutual_info_score

def _mutual_info_seqs(act_seqs, var_vals, nA):
    """
    Estimates the mutual information between human actions with a discrete variable naively,
    by associating each unique sequence of actions with a natural number. We then compute the 
    mutual information between the trajectory labels and the discrete variable.
    
    Args:
        act_seqs (list-like): a list of human action sequences, encoded as lists of discrete 
                            actions, with values in (0,...,nA-1).
        var_vals (list-like): a list of values of a discrete variable, where the ith entry
                            corresponds to the value of this variable at the ith trajectory. 
        nA: the number of actions the human can take. Must be larger than every action. 
    
    Returns:
        (float): the mutual information (in bits) between the human acts and the discrete variable.
    
    """
    encoded_act_seqs = []
    for act_seq in act_seqs:
        encoded_act_seq = 0
        for a in act_seq:
            encoded_act_seq = encoded_act_seq*(nA+1) + (a+1)
        encoded_act_seqs.append(encoded_act_seq)

    #TODO: write my own mutual_info_score so as to not depend on sklearn
    return mutual_info_score(encoded_act_seqs, var_vals)/np.log(2) #need log(2) to convert to bits

def mutual_info_acts_best_arm(trajs, n_arms, max_length=None):
    """
    Estimates the mutual information (in bits) between the human actions and 
    the best arm for the given trajectories in the bandit environment. 
    
    
    Args:
        trajs (list): a list of trajectory dictionaries, where traj['acts'] returns the 
                       actions taken by the human along this trajectory, and traj['best_arm']
                       returns the best arm of the environment the trajectory is generated
        n_arms (int): the number of arms in the bandit environment that generated these 
                        trajectories.
        max_length(int): the maximum length of trajectories to consider. Trajectories longer
                        than max_length will be truncated to this length. 
       
    Returns:
        (float): the mutual information between the human actions (up to time max_length)
                and the best arm, in bits. This in the range [0, log(bandit_env.nA)]
    """
    act_seqs_truncated = [traj["acts"][0:max_length] for traj in trajs]
    best_arms = [traj["best_arm"] for traj in trajs]
            
    return _mutual_info_seqs(act_seqs_truncated, best_arms, n_arms)

def _frequency_agreement(act_seqs1, act_seqs2, max_length=None):
    """
    Computes the frequency that two act sequences agree at each time.

    Args:
        act_seqs1 (list-like): a list of action sequences, encoded as lists of discrete 
                            actions, with values in (0,...,nA-1).
        act_seqs2 (list-like): a list of action sequences, encoded as lists of discrete 
                            actions, with values in (0,...,nA-1).
    
    Returns:
        (list): a list of floats of length max_length, indicating the frequency of agreement
        at each time.
    """

    if max_length == None:
        max_length = max((len(act_seq) for act_seq in act_seqs1))
        max_length = max(max_length, max((len(act_seq) for act_seq in act_seqs2)))

    agreement_counts = [0 for t in range(max_length)]
    total_counts = [0 for t in range(max_length)]

    for act_seq1, act_seq2 in zip(act_seqs1, act_seqs2):
        for t, (a_H1, a_H2) in enumerate(zip(act_seq1, act_seq2)):
            total_counts[t] += 1
            agreement_counts[t] += (a_H1 == a_H2)

    frequencies = [agreement_counts[t]/total_counts[t] for t in range(max_length)]
    return frequencies


def frequency_best_arm(trajs, n_arms, max_length=None):
    """
    Computes the frequency of selecting the best arm at each timestep in these trajectories.
    
    Really a wrapper for _frequency_agreement. 

    Args:
        trajs (list): a list of trajectory dictionaries, where traj['acts'] returns the 
                       actions taken by the human along this trajectory, and traj['best_arm']
                       returns the best arm of the environment the trajectory is generated
        n_arms (int): the number of arms in the bandit environment that generated these 
                        trajectories.
        max_length(int): the maximum length of trajectories to consider. Trajectories longer
                        than max_length will be truncated to this length. 

    Returns:
        (list): a list of floats of length max_length, indicating the frequency the best arm is 
        selected at each time.
    """
    if max_length == None:
        max_length = max((len(traj["acts"]) for traj in trajs))

    act_seqs_truncated = [traj["acts"][0:max_length] for traj in trajs]
    best_arms = [[traj["best_arm"] for i in range(max_length)] for traj in trajs]
            
    return _frequency_agreement(act_seqs_truncated, best_arms, max_length)



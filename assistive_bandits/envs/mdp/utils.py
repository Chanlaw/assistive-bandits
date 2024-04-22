from __future__ import print_function
import numpy as np

from gym.envs.toy_text.discrete import DiscreteEnv
from crl.envs.mdp.frozen_lake2 import FrozenLakeEnv

#DiscreteEnv is literally a discrete MDP. 

def value_iteration(mdp, gamma, max_iter, tol=1e-5, debug=False):
    """
    Computes the value of each state in, as well as the optimal policy for, the given MDP 
        using Value Iteration.

    args:
        mdp (DiscreteEnv): the MDP to solve. 
        gamma (DiscreteEnv): the discount rate
        max_iter
    """

    if debug:
        print("Iteration | max|V-Vprev| | # chg actions | V[12]")
        print("----------+--------------+---------------+---------")

    V = np.zeros(mdp.nS)
    pi = None

    for i in range(max_iter):
        new_V = np.zeros(mdp.nS, dtype=np.float32)
        new_pi = np.zeros(mdp.nS, dtype=np.int16)

        for s in mdp.P.keys():
            val = []
            for _, outcome in mdp.P[s].items():
                val.append(sum([P*(R + gamma*V[s_prime]) for P, s_prime, R, done in outcome]))
            new_pi[s]=(np.argmax(val))
            new_V[s]=(np.max(val))

        max_diff = np.abs(new_V - V).max()

        V = new_V
        pi = new_pi
        if debug:
            nChgActions="N/A" if i is None else (pi != new_pi).sum()
            print("%4i      | %6.5f      | %4s          | %5.3f"%(i, max_diff, nChgActions, V[12]))

        if max_diff < tol:
            break

    return V, pi

def compute_vpi(pi, mdp, gamma):
    """
    Computes the state value function associated with the given policy
    """
    P = np.zeros(shape=(mdp.nS, mdp.nS))
    R = np.zeros(shape=(mdp.nS))
    # Recover P, R from mdp
    for s in range(mdp.nS):
        for (Pr, s_prime, r, d) in mdp.P[s][pi[s]]:
            P[s][s_prime] += Pr
            R[s] += Pr * r
    V = np.linalg.solve(np.identity(mdp.nS) - gamma * P, R)
    return V

def compute_qpi(vpi, mdp,  gamma):
    """
    Computes state-action value function associated with the given (state) value function.
    """
    Qpi = np.zeros((mdp.nS, mdp.nA))
    for s in range(mdp.nS):
            for a, outcome in mdp.P[s].items():
                Qpi[s][a] = sum([P*(R + gamma*vpi[s_prime]) for P, s_prime, R, d in outcome])
    return Qpi

def policy_iteration(mdp, gamma, max_iter, debug=False):
    
    pi_prev = np.zeros(mdp.nS,dtype='int')
    if debug:
        print("Iteration | # chg actions | V[12]")
        print("----------+---------------+---------")
    for i in range(max_iter):        
        vpi = compute_vpi(pi_prev, mdp, gamma)
        qpi = compute_qpi(vpi, mdp, gamma)
        pi = qpi.argmax(axis=1)
        nchanged = (pi != pi_prev).sum()
        if debug:
            print("%4i      | %6i        | %6.5f"%(i, nchanged, vpi[12]))
        pi_prev = pi
        if nchanged == 0:
            vpi = compute_vpi(pi_prev, mdp, gamma)
            break
    return vpi, pi

if __name__ == "__main__":
    import time
    w_env = env = FrozenLakeEnv()
    V, pi = value_iteration(env, 0.9, 100, debug=True)

    V2, pi2 = policy_iteration(env, 0.9, 100, debug=True)
    print(V.reshape(env.nrow, env.ncol))
    print(V2.reshape(env.nrow, env.ncol))
    print(pi.reshape(env.nrow, env.ncol))
    print(pi2.reshape(env.nrow, env.ncol))
    import matplotlib.pyplot as plt
    feature_sums = np.array([np.sum(w_env.feats[s]) for s in range(w_env.nS)])
    feature_sums = feature_sums.reshape((w_env.nrow, w_env.ncol))
    # fig, ax = plt.subplots()
    # heatmap = ax.pcolor(feature_sums, vmin=0., vmax=np.max(feature_sums))
    # plt.colorbar(heatmap)

    # plt.show()

    state_rewards=np.array([np.dot(w_env.feats[s], w_env.theta) for s in range(w_env.nS)])
    state_rewards = state_rewards.reshape((w_env.nrow, w_env.ncol))

    fig, ax = plt.subplots()
    heatmap = ax.pcolor(state_rewards, vmin=-2*np.max(feature_sums), vmax=2*np.max(feature_sums))
    plt.colorbar(heatmap)
    plt.gca().invert_yaxis()

    plt.show()

    V = V.reshape((env.nrow, env.ncol))
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(V, vmin=np.min(V), vmax=np.max(V))
    plt.colorbar(heatmap)
    plt.gca().invert_yaxis()
    plt.show()

    # start_time=time.time()
    # for i in range(100):
    #     env.reset()
    #     policy_iteration(env, 0.9, 100, debug=False)

    # print("Total elapsed time: {} seconds".format(time.time() - start_time))

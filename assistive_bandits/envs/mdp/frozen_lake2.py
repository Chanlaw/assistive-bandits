from __future__ import print_function, division
import sys

import numpy as np

from gym import utils
from crl.envs.mdp.feature_mdp import FeatureMDP



MAPS = {
    "5x5": [
        "IFFFI",
        "FFFFF",
        "FFSFF",
        "FFFFF",
        "IFFFI"
    ],
    "9x9": [
        "FFFFFFFFF",
        "FIFFFFFIF",
        "FFFFFFFFF",
        "FFFFFFFFF",
        "FFFFSFFFF",
        "FFFFFFFFF",
        "FFFFFFFFF",
        "FIFFFFFIF",
        "FFFFFFFFF"
    ],
}

LEFT = 0
DOWN  = 1
RIGHT = 2
UP = 3
STAY = 4

class FrozenLakeEnv(FeatureMDP):
    """
    With apologies to the original frozenlakeenv

    S: starting point, safe
    F: frozen surface, safe
    I: point of interest

    Features are of the form [is_hole, w_0, w_1, ...] where w_0 is some measure inversely correlated to 
        distance to the 0th warmth source
    """
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc=None, map_name="5x5", is_slippery=True, horizon=20, theta=None, 
                    random_theta=True, warmth_decay="linear", theta_dist="hypercube"):

        if theta is None and not random_theta:
            raise ValueError("Theta can't be none if random_theta is false")
        if desc is None and map_name is None:
            raise ValueError('Must provide either desc or map_name')
        elif desc is None:
            desc = MAPS[map_name]

        if warmth_decay not in ["linear", "inverse exponential", "inverse quadratic"]:
            raise ValueError
        if theta_dist not in ["hypercube", "uniform"]:
            raise ValueError

        self.theta_dist = theta_dist
        self.warmth_decay = warmth_decay

        self.desc =  desc = np.asarray(desc, dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape

        def inc(row, col, a):
            if a==0: # left
                col = max(col-1,0)
            elif a==1: # down
                row = min(row+1,nrow-1)
            elif a==2: # right
                col = min(col+1,ncol-1)
            elif a==3: # up
                row = max(row-1,0)
            return (row, col)

        nA = 5
        nS = nrow * ncol


        self.warmth_sources = [(i, j) for i, row in enumerate(desc) \
                                for j, x in enumerate(row) if x==b'I']
        self.n_warmth_sources = len(self.warmth_sources)
        if theta is None:
            self.theta = theta = np.random.rand(self.n_warmth_sources)*4-2

        self.random_theta = random_theta
        self.theta = theta
        self.horizon = horizon

        feats = []

        T = {}
        starting_states = []

        for i, row in enumerate(desc):
            for j, x in enumerate(row):
                if warmth_decay=="inverse quadratic":
                    state_feat = [1./(1+np.linalg.norm(np.array([i-c_i,j-c_j]))**2) \
                                                for c_i, c_j in self.warmth_sources] 
                elif warmth_decay=="inverse exponential":
                    state_feat = [np.exp(-np.linalg.norm(np.array([i-c_i,j-c_j]))) \
                                                for c_i, c_j in self.warmth_sources] 
                elif warmth_decay=="linear":
                    norm_constant = (self.nrow + self.ncol)/2
                    state_feat = [np.linalg.norm(np.array([i-c_i,j-c_j]))/norm_constant
                                                for c_i, c_j in self.warmth_sources]
                starting_states.append(float(x==b'S'))
                state_feat = np.array(state_feat)
                feats.append(state_feat)
                s = self.state_from_tuple((i, j))
                T[s] = {}
                for a in range(5):
                    if is_slippery:
                        if a==STAY:
                            T[s][a] = [(1.0, s, False)]
                        else:
                            new_i, new_j = inc(i, j, a)
                            new_s = self.state_from_tuple((new_i,new_j))
                            T[s][a] = [(0.8, new_s, False)]
                            for b in [(a-1)%4, (a+1)%4]:
                                new_i, new_j = inc(new_i, new_j, b)
                                new_s = self.state_from_tuple((new_i,new_j))
                                T[s][a].append((0.1, new_s, False))
                    else:
                        new_i, new_j = inc(i, j, a)
                        new_s = self.state_from_tuple((new_i,new_j))
                        T[s][a] = [(1.0, new_s, False)]
        isd = np.array(starting_states)/sum(starting_states)
        self.T = T
        super(FrozenLakeEnv, self).__init__(nS, nA, T, theta, isd, feats)

    def state_to_tuple(self, obs):
        row = obs//self.ncol
        col = obs%self.ncol
        return row, col

    def state_from_tuple(self, obs):
        row, col = obs
        return row * self.ncol + col

    def set_theta(self, theta):
        P = {}
        for s in self.T:
            P[s] = {}
            for a in self.T[s]:
                P[s][a] = []
                for prob, next_s, done in self.T[s][a]:
                    rew = np.dot(self.feats[s], theta)
                    P[s][a].append((prob, next_s, rew, done))
        self.P = P
        self.theta = theta


    def _reset(self):
        self.n_timesteps = 0
        self.accumulated_rew = 0.
        self.accumulated_regret = 0.
        if self.random_theta:
            if self.theta_dist=="hypercube":
                theta = self.np_random.binomial(1, 0.5, size=self.n_warmth_sources)*2 -1
            elif self.theta_dist=="uniform":
                theta = self.np_random.rand(self.n_warmth_sources)*4-2
            self.set_theta(theta)
        self.state_rewards=np.array([np.dot(self.feats[s], self.theta) for s in range(self.nS)])
        self.best_reward=np.max(self.state_rewards)

        return super(FrozenLakeEnv, self)._reset()


    def _step(self, a):
        s, r, d, info = super(FrozenLakeEnv, self)._step(a)
        self.accumulated_rew += r
        self.accumulated_regret += self.best_reward - r
        self.n_timesteps += 1
        d = (self.n_timesteps >= self.horizon)
        info["accumulated reward"] = self.accumulated_rew
        info["best reward"] = self.best_reward
        info["regret"] = self.best_reward - r
        info["accumulated regret"] = self.accumulated_regret
        return (s, r, d, info)


    def _render(self, mode='human', close=False):
        if close:
            return
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Left","Down","Right","Up", "Stay"][self.lastaction]))
            outfile.write("accumulated reward: {}\n".format(self.accumulated_rew))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        if mode != 'human':
            return outfile

if __name__ == "__main__":
    env = FrozenLakeEnv()
    print(env.feats)

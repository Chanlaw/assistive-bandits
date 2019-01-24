from __future__ import division, print_function

import time
import numpy as np
from utils import categorical_sample
from tqdm import tqdm

class SearchNode(object):
    """ Represents a single node in a search tree. """
    NODE_REGISTER = {}
    def __init__(self, state, env, gamma, done, node_register={}):
        self.state = state
        self.env = env
        self.gamma = gamma
        self.done = done
        self.node_register = node_register
        
        if not done:
            self.value = env.whittle_integral(state, discount=0.95)
        else:
            self.value = 0
        self.successors = env.transitions(state)
        self.solved = False
        self.node_register[state] = self
        
    def q_value(self, a):
        if self.done:
            return 0
        Q = 0
        for p, n_s, rew, done in self.successors[a]:
            if n_s not in self.node_register:
                n_s = SearchNode(n_s, self.env, self.gamma, done, self.node_register)
            else:
                n_s = self.node_register[n_s]
            Q += p * (rew + self.gamma * n_s.value)
        return Q
    
    def greedy_a(self):
        best_q = -np.inf
        for a in self.successors:
            curr_q = self.q_value(a)
            if curr_q > best_q:
                best_q = curr_q
                best_a = a
        return best_a
    
    def update(self):
        a = self.greedy_a()
        self.value = self.q_value(a)
    
    def sample_next(self, a):
        transitions = self.successors[a]
        i = categorical_sample([t[0] for t in transitions])
        p, n_s, rew, done = transitions[i]
        if n_s not in self.node_register:
            n_s = SearchNode(n_s, self.env, self.gamma, done, self.node_register)
        else:
            n_s = self.node_register[n_s]
        return n_s
    
    def residual(self):
        a = self.greedy_a()
        return np.abs(self.value - self.q_value(a))  
    
    def check_solved(self, tol):
        converged = True
        _open = []
        _closed = []
        if not self.solved: _open.append(self)
        while _open:
            s = _open.pop()
            _closed.append(s)
            if s.residual() > tol:
                converged = False
                continue
            a = s.greedy_a()
            for p, n_s, rew, done in s.successors[a]:
                # note that n_s is guaranteed to be in the register due to greedy_a()
                n_s = self.node_register[n_s] 
                if not n_s.solved and (not (n_s in _open) or (n_s in _closed)):
                    _open.append(n_s)
        if converged:
            # print "found converged states"
            for s in _closed:
                s.solved=True
        else:
            for s in _closed:
                s.update()
        return converged

def LRTDP(env, start_state, gamma, tol=1e-7, max_iter=1000, node_register={}, max_t = None):
    if gamma < 1:
        sim_len = int(np.ceil(np.log(tol) / np.log(gamma)))
    else:
        sim_len = env.max_timesteps

    # SearchNode.NODE_REGISTER = {}
    if max_t is not None:
        start = time.time()
    
    root = SearchNode(start_state, env, gamma, False, node_register)
    for _ in tqdm(range(max_iter)):
        if max_t is not None and time.time() - start > max_t:
            break
        if root.solved:
            break
        visited = []
        s = root
        for i in range(sim_len):
            visited.append(s)
            a = s.greedy_a()
            s = s.sample_next(a)
            if s.solved or s.done:
                break
        while visited:
            s = visited[-1]
            visited = visited[:-1]
            if not s.check_solved(tol):
                break

    return root, root.solved, node_register
                
        
from __future__ import print_function, division

class Model(object):
    def __init__(self):
        return
    
    def generate_state_particles(self, n_particles):
        raise NotImplementedError

    def sample_transition(self, state, act):
        raise NotImplementedError
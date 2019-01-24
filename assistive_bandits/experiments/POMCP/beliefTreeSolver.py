from __future__ import print_function, division

class beliefTreeSolver(object):
    """ Abstract Solver Class for Belief Trees. """
    def __init__(self, model):
        self.model = model
        raise NotImplementedError
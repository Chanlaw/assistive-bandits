from __future__ import print_function, division

import numpy as np
import tensorflow as tf

from gym.spaces import Box
from rllab.spaces.base import Space
from rllab.misc import special
from rllab.misc import ext

class Box(Box, Space):
    """
    Overrides gym.spaces.box.Box. This is just a box in R^n.
    """

    def __init__(self, low, high, shape=None):
        """
        Two kinds of valid input:
            Box(-1.0, 1.0, (3,4)) # low and high are scalars, and shape is provided
            Box(np.array([-1.0,-2.0]), np.array([2.0,4.0])) # low and high are arrays of the same shape
        """
        super(Box, self).__init__(low, high, shape=shape)
        self._flat_dim = np.prod(self.low.shape)

    def flatten(self, x):
        return np.array(x)
    
    def flatten_n(self,x):
        return np.array(x)
    
    def unflatten(self, flat_x):
        return np.array(flat_x)
    
    def unflatten_n(self, flat_x):
        return np.array(flat_x)

    def new_tensor_variable(self, name, extra_dims):
        # needed for safe conversion to float32
        return tf.placeholder(dtype=tf.uint8, shape=[None] * extra_dims + [self.flat_dim], name=name)
    
    @property
    def shape(self):
        return self.low.shape

    @property
    def flat_dim(self):
        return self._flat_dim

    @property 
    def n(self):
        return self._flat_dim

    @property
    def dtype(self):
        return tf.uint8

    def __hash__(self):
        return hash(self.low.tostring()) + hash(self.high.tostring())

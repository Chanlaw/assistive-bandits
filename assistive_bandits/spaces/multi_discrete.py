from __future__ import print_function, division

import numpy as np
import tensorflow as tf

from gym.spaces.multi_discrete import MultiDiscrete
from rllab.spaces.base import Space
from rllab.misc import special
from rllab.misc import ext

class MultiDiscrete(MultiDiscrete, Space):
    """
    Overrides gym.spaces.multi_discrete.MultiDiscrete. This space is a series of discrete spaces 
    with different parameters.
    """

    def __init__(self, array_of_param_array):
        """
        Initialize the MultiDiscrete space. 

        Args:
            array_of_param_array (array-like): array of arrays of [min, max] for each discrete space. 
                                                Note that both min and max are inclusive.
        """
        super(MultiDiscrete, self).__init__(array_of_param_array)
        self._flat_dim = np.sum(self.high - self.low + 1)
        self._discrete_dim = np.prod(self.high - self.low  + 1)

    def to_discrete(self, x):
        discrete_x = 0
        for xi, increment in zip(x, (self.high - self.low + 1)):
            discrete_x *= increment
            discrete_x += xi
        return discrete_x
    
    def to_discrete_n(self, x):
        raise NotImplementedError
        
    def from_discrete(self, discrete_x):
        x = []
        for increment in reversed((self.high - self.low + 1)):
            xi = discrete_x % increment
            discrete_x = discrete_x // increment
            x.append(xi)
        x.reverse()
        return np.array(x)
        
    def from_discrete_n(self, discrete_x):
        raise NotImplementedError
        
    def flatten(self, x):
        assert len(x) == self.shape
        flat_x = np.zeros(self.flat_dim)
        offset = 0 
        for i in range(self.shape):
            entry = x[i]-self.low[i]
            flat_x[offset+entry] = 1
            offset += self.high[i] - self.low[i] + 1
        return flat_x
    
    def flatten_n(self,x):
        assert len(x[0]) == self.shape
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        flat_x = np.zeros([len(x), self.flat_dim])
        offset=0
        for i in range(self.shape):
            entry = x[:,i] - self.low[i]
            flat_x[np.arange(len(x)), offset+entry] = 1
            offset += self.high[i] - self.low[i] + 1
        return flat_x
    
    def unflatten(self, flat_x):
        assert len(flat_x) == self.flat_dim
        assert sum(flat_x) == self.shape
        x, = np.nonzero(flat_x)
        offset = 0
        for i in range(self.shape):
            x[i] -= offset - self.low[i]
            offset += self.high[i] - self.low[i] + 1
        return x
    
    def unflatten_n(self, flat_x):
        if len(flat_x)==0:
            return flat_x
        assert len(flat_x[0]) == self.flat_dim
        assert sum(flat_x[0]) == self.shape
        _, x = np.nonzero(flat_x)
        offset = 0
        x = np.reshape(x, [len(flat_x), self.shape])
        for i in range(self.shape):
            x[:,i] -= offset - self.low[i]
            offset += self.high[i] - self.low[i] + 1
        return x
        
    def __repr__(self):
        return "MultiDiscrete" + str(self.num_discrete_space)

    def __eq__(self, other):
        if not isinstance(other, MultiDiscrete):
            return False
        return np.array_equal(self.low, other.low) and np.array_equal(self.high, other.high)

    def new_tensor_variable(self, name, extra_dims):
        # needed for safe conversion to float32
        return tf.placeholder(dtype=tf.uint8, shape=[None] * extra_dims + [self.flat_dim], name=name)
    
    @property
    def shape(self):
        return self.num_discrete_space

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

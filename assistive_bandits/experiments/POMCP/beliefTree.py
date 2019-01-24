from __future__ import print_function, division

import numpy as np

class BeliefNode(object):
    """
    Represents a single node in a (discrete) belief tree.

    Attributes:
        state_particles (list): a set of state particles representing the belief state. 
        act: the last action associated with this history.
        obs: the last observation associated with this history.
        action_map (ActionMap): the ActionMap associated with this node, which stores the actions taken from
                this history, as well as their associated statistics. 
        depth: the depth of this node in the belief true
        data: extra data associated with this belief node. 
        parent_node: the parent belief node of this node. 
    """

    def __init__(self, act, obs, n_actions, data=None, parent_node=None):
        """
        Initialize a belief node.

        Args:
            act (hashable): the last action associated with this history
            obs (hashable): the last observation associated with this history
            data: any additional data associated with this belief node. 
            parent_node (BeliefNode): the parent BeliefNode (if any)
        """
        self.data = data
        self.act = act
        self.obs = obs
        self.n_actions = n_actions
        self.action_map = ActionMap(self, n_actions)
        self.state_particles = []

        if parent_node is not None:
            self.parent_node = parent_node
            # Correctly calculate the depth based on the parent node.
            self.depth = self.parent_node.depth + 1
        else:
            self.parent_node = None
            self.depth = 0

    def sample_particle(self, np_random=None):
        """
        Sample a state particle randomly. 

        Args:
            np_random (np.random.RandomState): a RandomState. If specified, use it to sample the 
                                        particle. Otherwise, sample using np.random.choice.
        """
        if np_random is not None:
            np_random.choice(self.state_particles)
        return np.random.choice(self.state_particles)

    def copy(self):
        """ Makes a copy of the belief node. """
        bn = BeliefNode(self.act, self.obs, self.data.copy(), self.parent_node)
        # share a reference to the action map and state particles
        bn.action_map = self.action_map
        bn.state_particles = self.state_particles
        return bn

    def get_last_observation(self):
        """ Returns the last observation before this node. """
        return self.parent_node.obs

    def get_last_action(self):
        """ Returns the last observation before this node. """
        return self.parent_node.act

    def get_child(self, act, obs):
        """ Returns the child belief node associated with the action observation pair"""
        obs_map = self.action_map.entries[act]
        if obs_map:
            return obs_map[obs]
        return none

    def create_or_get_child(self, act, obs, n_actions=None, data=None):
        """ 
        Creates or gets the child belief node associated with the action observation pair.

        Returns:
            (BeliefNode): the child node associated with the action observation pair. 
            (bool): whether the child node was created as a result of this function call
        """
        obs_map = self.action_map.get_obs_map(act)
        added = False
        if obs_map is None:
            obs_map = {}
            self.action_map.set_obs_map(act, obs_map)
        if obs not in obs_map:
            if n_actions is None:
                child = BeliefNode(act, obs, self.action_map.n_actions, data=data, parent_node=self)
            else:
                child = BeliefNode(act, obs, n_actions, data=data, parent_node=self)
            obs_map[obs] = child
            added = True
        else:
            child = obs_map[obs]
        return child, added

class ActionMap(object):
    """
    Stores the actions that have been taken from a given BeliefNode, as well as the 
    statistics and child nodes associated with them.

    Attributes:
        n_actions (int): the number of actions that can be taken from the BeliefNode
        entries: a list mapping actions to dicts mapping observations to BeliefNodes
        visit_counts: a list mapping actions to the number of times 
        total_visit_count: total number of times actions in this map have been visited. 
        stats: a list mapping actions to dicts of stats associated with the actions. 
    """
    def __init__(self, owner, n_actions):
        """
        Initializes this action map. 
        """
        self.n_actions = n_actions
        self.entries = [None for _ in range(n_actions)]
        self.visit_counts = [0 for _ in range(n_actions)]
        self.total_visit_count = 0
        self.stats = [None for _ in range(n_actions)]

    def reset(self):
        self.entries = [None for _ in range(self.n_actions)]
        self.visit_counts = [0 for _ in range(self.n_actions)]
        self.total_visit_count = 0
        self.stats = [None for _ in range(self.n_actions)]

    def get_child_nodes(self):
        """ Returns a list of child nodes associated with this map. """
        children = []
        for entry in self.entries:
            if entry is not None:
                children += entry.values()
        return children

    def get_obs_map(self, act):
        """ get the dict mapping observations to belief nodes corresponding to this action. """
        return self.entries[act]

    def set_obs_map(self, act, obs_map):
        self.entries[act] = obs_map

class BeliefTree(object):
    """
    Represents an entire belief tree. The key functionality in this class are the pruning methods. 
    """
    def __init__(self, model, n_actions, obs=None):
        self.model = model
        self.root = BeliefNode(None, obs, n_actions)
        self.n_actions=n_actions

    def reset(self, obs=None):
        """ Resets the tree, returning the new root belief node. """
        self.prune_tree()
        self.root = BeliefNode(None, obs, self.n_actions)
        return self.root

    def prune_tree(self):
        """ Clears out a belief tree. """
        self.prune_node(self.root)
        self.root = None

    def prune_node(self, bn):
        """ Recursively removes the node bn and all of its descendants from the belief tree. """
        if bn is None:
            return
        if bn.parent_node:
            bn.parent_node.action_map.entries[bn.act] = None
            bn.parent_node = None
        bn.action_map.owner=None
        children = bn.action_map.get_child_nodes()

        for node in children:
            self.prune_node(node)
            node = None
        bn.action_map.reset()
        bn.action_map = None
        bn.state_particles = []

    def prune_siblings(self, bn):
        """ Recursively removes the sibling nodes and their descendants from the tree. """
        if bn is None:
            return
        parent = bn.parent_node

        if parent is not None:
            for child in parent.action_map.get_child_nodes():
                if child is not bn:
                    self.prune_node(child)

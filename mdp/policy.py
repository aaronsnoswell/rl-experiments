"""
Defines an interface for a Markov Decision Process policy
"""

import numpy as np


class Policy():
    """
    A policy is a distribution over actions, given the current state
    """

    def __init__(self):
        """
        Constructor
        """
        raise NotImplementedError


    def __str__(self):
        """
        Get string representation
        """
        return "<Policy object>"


    def get_action_distribution(self, current_state):
        """
        Returns a distribution {a_i:p_i, ... a_n:p_n} over action
        given the current state
        """
        return self.policy_mapping[current_state]


    def get_action(self, mdp, current_state):
        """
        Returns a sampled action from the given state
        """

        action_distribution = self.get_action_distribution(current_state)

        action_weights = np.array([])
        for action in mdp.get_action_set():
            action_weights = np.append(action_weights, action_distribution.get(action, 0))

        return np.random.choice(mdp.get_action_set(), p=action_weights)


class UniformRandomPolicy(Policy):
    """
    Implements a uniform random distribution over possible actions
    from each state
    """


    def __init__(self, mdp):
        """
        Constructor
        """

        # Store reference to MDP type
        self.mdp_type = type(mdp)

        self.policy_mapping = {}
        
        for state in mdp.get_state_set():

            self.policy_mapping[state] = {}

            # Initialize all actions to 0 preference
            for action in mdp.get_action_set():
                self.policy_mapping[state][action] = 0

            # Apply a uniform distribution to the possible actions
            possible_actions = mdp.get_possible_action_mapping()[state]
            for action in possible_actions:
                self.policy_mapping[state][action] = 1.0 / len(possible_actions)


    def __str__(self):
        """
        Get string representation
        """
        return "<UniformRandomPolicy initialized on {} MarkovDecisionProcess>".format(
            self.mdp_type
        )

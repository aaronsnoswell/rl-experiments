"""
Defines an interface for a Markov Decision Process policy
"""


class Policy():
    """
    A policy is a distribution over actions, given the current state
    """

    def __init__(self):
        """
        Constructor
        """
        raise NotImplementedError

    def get_action_distribution(current_state):
        """
        Returns a distribution {p_i: a_i, ... p_n: a_n} over action
        given the current state
        """
        raise NotImplementedError

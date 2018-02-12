"""
An implementation of student markov chain from David Silver's Reinforcement Learning
lecture series, lecture 2
"""

import numpy as np

from mdp import MarkovProcess


class Student(MarkovProcess):
    """
    An implementation of student markov chain from David Silver's
    Reinforcement Learning lecture series, lecture 2
    """

    def __init__(self):
        """
        Constructor
        """
        
        self.state_set, self.state_transition_matrix = MarkovProcess.from_dict(
            {
                "C1": {
                    "FB": 0.5,
                    "C2": 0.5
                },
                "C2": {
                    "Sleep": 0.2,
                    "C3": 0.8
                },
                "C3": {
                    "Pass": 0.6,
                    "Pub": 0.4
                },
                "Pass": {
                    "Sleep": 1
                },
                "Pub": {
                    "C1": 0.2,
                    "C2": 0.4,
                    "C3": 0.4
                },
                "FB": {
                    "FB": 0.9,
                    "C1": 0.1
                },
            }
        )


    def get_state_set():
        """
        Returns the set of states
        """
        return self.state_set


    def get_state_transition_matrix():
        """
        Returns the state transition matrix
        """
        return self.transition_matrix


a = Student()

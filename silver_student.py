"""
An implementation of student markov chain from David Silver's Reinforcement Learning
lecture series, lecture 2
"""

import numpy as np

from mdp import MarkovProcess
from mdp import MarkovRewardProcess


class Student(MarkovRewardProcess):
    """
    An implementation of student markov chain from David Silver's
    Reinforcement Learning lecture series, lecture 2
    """

    def __init__(self):
        """
        Constructor
        """

        self.state_set, self.transition_matrix, self.terminal_state_set = MarkovProcess.from_dict(
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

        self.reward_mapping = {
            "C1": -2,
            "C2": -2,
            "C3": -2,
            "Pass": 10,
            "Pub": 1,
            "FB": -1,
            "Sleep": 0
        }

        self.discount_factor = 0.5


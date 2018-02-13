"""
An implementation of student markov chain from David Silver's Reinforcement Learning
lecture series, lecture 2
"""

import numpy as np

from mdp import MarkovDecisionProcess
from mdp import UniformRandomPolicy


class Student(MarkovDecisionProcess):
    """
    An implementation of student markov chain from David Silver's
    Reinforcement Learning lecture series, lecture 2
    """

    def __init__(self):
        """
        Constructor
        """

        self.state_set, self.action_set, self.transition_matrix, \
            self.possible_action_mapping, self.terminal_state_set \
                = MarkovDecisionProcess.from_dict(
            {
                "C1": {
                    "Study": {
                        "C2": 1
                    },
                    "Facebook": {
                        "FB": 1
                    }
                },
                "C2": {
                    "Sleep": {
                        "Sleep": 1
                    },
                    "Study": {
                        "C3": 1
                    }
                },
                "C3": {
                    "Study": {
                        "Sleep": 1
                    },
                    "Pub": {
                        "C1": 0.2,
                        "C2": 0.4,
                        "C3": 0.4
                    }
                },
                "FB": {
                    "Facebook": {
                        "FB": 1
                    },
                    "Quit": {
                        "C1": 1
                    }
                },
            }
        )

        self.reward_mapping = {
            "C1": {
                "Study": -2,
                "Facebook": -1
            },
            "C2": {
                "Study": -2,
                "Sleep": 0
            },
            "C3": {
                "Study": 10,
                "Pub": 1
            },
            "FB": {
                "Facebook": -1,
                "Quit": 0
            }
        }

        self.discount_factor = 0.5


aaron = Student()
print(aaron)
pol = UniformRandomPolicy(aaron)
print(aaron.get_value_map(pol))
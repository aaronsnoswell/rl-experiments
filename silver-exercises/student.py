"""
An implementation of student markov chain from David Silver's Reinforcement Learning
lecture series, lecture 2
"""

import numpy as np

# Get the MDP directory in our path
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mdp import MarkovDecisionProcess
from mdp import UniformRandomPolicy, iterative_policy_evaluation


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


def main():
    """
    Excercises the Student() class
    """

    # Create a new student MDP instance
    jane = Student()
    print(jane)

    # And create a policy for them
    not_great_studier = UniformRandomPolicy(jane)

    # Test value map estimation under the policy
    print(jane.get_value_map(not_great_studier))

    # Decompose the MDP, under the policy, to a Markov Reward Process
    # I.e. remove the 'agency' from this Student (hence the zombie joke)
    jane_zombie = jane.decompose(not_great_studier)
    print(jane_zombie)

    # Decompose the MRP to a Markov Process
    # A Zombie without any sense of value is just a meaningless zombie
    # (Sorry for the terrible puns)
    jane_meaningless_zombie = jane_zombie.decompose()
    print(jane_meaningless_zombie)

    # And just for fun, compute the stationary distribution of the MP
    print(jane_meaningless_zombie.compute_stationary_distribution())
    # ... turns out Jane spends around 27% of her time on Facebook

if __name__ == "__main__":
    main()

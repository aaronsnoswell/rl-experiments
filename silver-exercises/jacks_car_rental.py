"""
An implementation of the jack's car rental problem from David Silver's RL
lecture series, lecture 3, p17
"""

import numpy as np
import matplotlib.pyplot as plt

# Get the MDP directory in our path
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import itertools
import math

from mdp import *


class JacksCarRental(MarkovDecisionProcess):
    """
    Class representing the Jack's Car Rentals MDP from David Silver's RL lectures
    """


    class State():
        """
        Container class representing a Jack's Car Rentals MDP state
        """

        def __init__(self, tup):
            """
            Constructor
            """

            # Number of cars at location 1 and 2
            self.l1_cars = tup[0]
            self.l2_cars = tup[1]


        def __str__(self):
            """
            String repr
            """
            return "({}, {})".format(self.l1_cars, self.l2_cars)


        def __repr__(self):
            """
            String repr
            """
            return "({}, {})".format(self.l1_cars, self.l2_cars)




    @staticmethod
    def poisson_prob(l, n):
        """
        Computes Poisson probability of getting n events given an expectation
        of l events
        """
        return l ** n / math.factorial(n) * math.exp(-1 * l)


    def __init__(
        self,
        *,
        max_cars=20,
        max_movement=5,
        average_hires=(3, 4),
        average_returns=(3 ,2),
        reward_per_hire=10
        ):
        """
        Constructor
        """

        # Max number of cars at location 1 or location 2
        self.max_cars = max_cars

        # Max number of cars you can move each night
        self.max_movement = max_movement

        # Average number of hires per day for (location 1, location 2)
        self.average_hires = average_hires

        # Average number of returns per day for (location 1, location 2)
        self.average_returns = average_returns

        # Reward for each hire ($)
        self.reward_per_hire = reward_per_hire

        # (Cars at location 1, Cars at location 2)
        self.state_set = np.array([JacksCarRental.State(t) for t in
            itertools.product(
                range(self.max_cars + 1),
                range(self.max_cars + 1)
            )
        ])

        # The Jack's car rentals MDP is unbounded (no terminal states)
        self.terminal_state_set = []
        
        # Number of cars moved overnight from location 1 to location 2
        self.action_set = np.array(
            list(
                range(
                    -self.max_movement,
                    self.max_movement + 1
                )
            )
        )

        # Transition matrix
        self.transition_matrix = np.zeros(
            shape = (
                len(self.state_set) * len(self.action_set),
                len(self.state_set)
            )
        )

        # Precompute the world probability transition grid (conditioned on actions)
        l1_world_probs = {}
        l2_world_probs = {}
        chance_moved_options = self.max_cars + self.max_movement
        chance_moved_range = range(-chance_moved_options, chance_moved_options+1)
        
        for chance_moved in chance_moved_range:

            l1_world_probs[chance_moved] = 0
            l2_world_probs[chance_moved] = 0

            for returns in range(self.max_cars):
                for hires in range(self.max_cars):
                    if returns - hires == chance_moved:

                        l1_world_probs[chance_moved] +=\
                            JacksCarRental.poisson_prob(self.average_returns[0], returns) *\
                            JacksCarRental.poisson_prob(self.average_hires[0], hires)

                        l2_world_probs[chance_moved] +=\
                            JacksCarRental.poisson_prob(self.average_returns[1], returns) *\
                            JacksCarRental.poisson_prob(self.average_hires[1], hires)


        # Populate the full transition matrix
        for csi, current_state in enumerate(self.state_set):

            for ai, action in enumerate(self.action_set):

                for nsi, new_state in enumerate(self.state_set):

                    # Determine the probability of the new state
                    # For this state to be valid, chance must have moved an
                    # appropriate number of cars to both l1 and l2
                    # As returns and hires are independent, and l1 and l2 are
                    # independent, we can multiply their probabilities
                    self.transition_matrix[csi * len(self.action_set) + ai, nsi] =\
                        l1_world_probs[new_state.l1_cars - current_state.l1_cars + action] *\
                        l2_world_probs[new_state.l2_cars - current_state.l2_cars - action]

        # Normalize the transition matrix rows to true probabilities
        for ri in range(self.transition_matrix.shape[0]):
            row = self.transition_matrix[ri, :]
            total = np.sum(row)
            self.transition_matrix[ri, :] = row / total


        # Populate the reward mapping
        self.reward_mapping = {}
        for csi, current_state in enumerate(self.state_set):

            current_state = current_state
            self.reward_mapping[current_state] = {}
            
            for ai, action in enumerate(self.action_set):

                self.reward_mapping[current_state][action] = 0

                for nsi, new_state in enumerate(self.state_set):

                    num_hires = current_state.l1_cars - new_state.l1_cars +\
                        current_state.l2_cars - new_state.l2_cars
                    num_hires = max(0, num_hires)

                    prob = self.transition_matrix[csi * len(self.action_set) + ai, nsi]

                    if num_hires != 0:
                        self.reward_mapping[current_state][action] +=\
                            num_hires * self.reward_per_hire * prob
                                


        self.possible_action_mapping = {}
        for state in self.state_set:
            self.possible_action_mapping[state] = self.action_set[
                (self.action_set <= state.l1_cars) & (-1 * self.action_set <= state.l2_cars)
            ]

        self.discount_factor = 0.5



def main():
    """
    Excercises the JacksCarRental() class
    """

    print("Initializing Jacks Car Rentals MDP")
    mdp = JacksCarRental()
    print("Done initializing")

    print("Constructing URP")
    urp = UniformRandomPolicy(mdp)
    print("Done constructing URP")



if __name__ == "__main__":
    main()

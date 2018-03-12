"""
An implementation of the Jack's Car Rental problem from Sutton and Barto,
Ex 4.2, p87
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
        reward_per_hire=10,
        cost_to_move=2,
        discount_factor=0.9
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

        # Cost to move a car ($)
        self.cost_to_move = cost_to_move

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

        # Possible action mapping
        self.possible_action_mapping = {}
        for state in self.state_set:
            self.possible_action_mapping[state] = self.action_set[
                (self.action_set <= state.l1_cars) & (-1 * self.action_set <= state.l2_cars)
            ]


        def _compute_prob_table(av_returns, av_hires):
            """
            Helper function to compute a poisson probability table given
            an average number of returns and hires
            """
            probs = {}
            expected_number_of_hires = {}
            for i in range(-self.max_cars - self.max_movement, self.max_cars + 1):
                
                probs[i] = 0
                expected_number_of_hires[i] = 0

                # p(i cars) is determined by a combination of returns and hires
                for returns in range(self.max_cars + 1):
                    prob_of_returns = JacksCarRental.poisson_prob(av_returns, returns)

                    for hires in range(self.max_cars +1):
                        prob_of_hires = JacksCarRental.poisson_prob(av_hires, returns)

                        if returns - hires != i: continue

                        # This was a valid combination of returns and hires
                        probs[i] += prob_of_returns * prob_of_hires
                        expected_number_of_hires[i] += prob_of_hires * hires
            
            return probs, expected_number_of_hires

        
        # Pre-compute the probabilities of certain numbers of cars arriving and location 1 and 2
        self._loc1_probs, self._loc1_hires = _compute_prob_table(self.average_returns[0], self.average_hires[0])
        self._loc2_probs, self._loc2_hires = _compute_prob_table(self.average_returns[1], self.average_hires[1])

        # Transition matrix
        self.transition_matrix = np.zeros(
            shape = (
                len(self.state_set) * len(self.action_set),
                len(self.state_set)
            )
        )

        # Reward function
        self.reward_mapping = {}

        # Build transition function and reward mapping
        for current_state in self.state_set:
            csi = np.where(self.state_set == current_state)[0][0]

            self.reward_mapping[current_state] = {}

            for action in self.possible_action_mapping[current_state]:
                asi = np.where(self.action_set == action)[0][0]
                
                # Figure out how many cars are where in the morning
                morning_car_counts = (
                    current_state.l1_cars - action,
                    current_state.l2_cars + action
                )

                # Loop over the transition matrix columns
                for possible_afternoon_state in self.state_set:
                    tsi = np.where(self.state_set == possible_afternoon_state)[0][0]

                    # To get to this afternoon state, chance needed to move cars around as follows
                    loc1_chance_moved = possible_afternoon_state.l1_cars - morning_car_counts[0]
                    loc2_chance_moved = possible_afternoon_state.l2_cars - morning_car_counts[1]

                    # As hires and returns at locations are independant, 
                    # p(possible_afternoon_state) = p(loc1_chance_moved) * p(loc2_chance_moved)
                    prob = self._loc1_probs[loc1_chance_moved] * self._loc2_probs[loc2_chance_moved]
                    self.transition_matrix[csi * len(self.action_set) + asi, tsi] = prob

                # Normalize the probabilites in this row of the transition matrix
                row_sum = sum(self.transition_matrix[csi * len(self.action_set) + asi, :])
                self.transition_matrix[csi * len(self.action_set) + asi, :] =\
                    self.transition_matrix[csi * len(self.action_set) + asi, :] / row_sum

                # Now compute the reward for this state-action pair
                # (must have first computed the full normalized transition matrix row)
                self.reward_mapping[current_state][action] = 0
                for possible_afternoon_state in self.state_set:
                    tsi = np.where(self.state_set == possible_afternoon_state)[0][0]

                    # To get to this afternoon state, chance needed to move cars around as follows
                    loc1_chance_moved = possible_afternoon_state.l1_cars - morning_car_counts[0]
                    loc2_chance_moved = possible_afternoon_state.l2_cars - morning_car_counts[1]

                    # Therefore the number of hires we had today must have been
                    num_hires = min(0, loc1_chance_moved) + min(0, loc2_chance_moved)

                    # This occured with the following probability
                    prob = self.transition_matrix[csi * len(self.action_set) + asi, tsi]

                    # So we can update the reward as follows
                    self.reward_mapping[current_state][action] += prob * num_hires

                # Convert the reward mapping entry (currently an
                # expected number of hires) to units of $
                self.reward_mapping[current_state][action] *= self.reward_per_hire

                # And finally, subtract the cost of moving the number of cars we moved
                self.reward_mapping[current_state][action] -= abs(action) * self.cost_to_move

        # Discount factor
        self.discount_factor = discount_factor


    def generate_contour_figure(self, policy):
        """
        Generate a contour plot of the given JCR policy
        """

        # Render settings
        line_width = 0.75
        line_color = "#dddddd"

        # Store width and height
        width = self.max_cars + 1
        height = self.max_cars + 1

        # Get render helpers
        fig = plt.gcf()
        ax = plt.gca()

        def color_from_action(ac):
            """
            Helper function to compute a greyscale color from an action
            """
            real_val = (ac + self.max_movement) / (self.max_movement * 2)
            color = [real_val] * 3
            return color


        # Compute pollicy grid
        policy_grid = np.empty(
            shape=(
                self.max_cars + 1,
                self.max_cars + 1
            ),
            dtype=int
        )
        for s in policy.policy_mapping:
            policy_grid[s.l1_cars, s.l2_cars] = policy.get_action(
                self,
                s,
                tie_breaker_action=0
            )

        # Draw policy
        for yi in range(height):
            for xi in range(width):

                render_pos = (
                    xi,
                    height - (yi + 1)
                )

                action = policy_grid[yi, xi]
                color = color_from_action(action)

                # Draw this square, and it's associated action as text
                ax.add_artist(plt.Rectangle(
                        render_pos,
                        width=1,
                        height=1,
                        color=color
                    )
                )
                draw_text(
                    xi,
                    yi,
                    height,
                    action,
                    textcolor=high_contrast_color(color),
                    formatstr="{: d}"
                )

        # Draw grid lines
        for i in range(height - 1):
            ax.add_artist(plt.Line2D(
                    (0, width),
                    (i+1, i+1),
                    color=line_color,
                    linewidth=line_width
                )
            )
        for i in range(width - 1):
            ax.add_artist(plt.Line2D(
                    (i+1, i+1),
                    (0, height),
                    color=line_color,
                    linewidth=line_width
                )
            )

        # Configure limits
        plt.xlim([0, width])
        plt.ylim([0, height])
        ax.set_aspect("equal", adjustable="box")

        # Configure labels
        plt.xticks(
            [i + 0.5 for i in range(0, self.max_cars + 1)],
            range(0, self.max_cars + 1)
        )
        plt.xlabel("Cars at location 2")
        plt.ylabel("Cars at location 1")
        plt.yticks(
            [i + 0.5 for i in range(0, self.max_cars + 1)],
            range(0, self.max_cars + 1)
        )
        ax.tick_params(length=0)


def main():
    """
    Excercises the JacksCarRental() class
    """

    print("Initializing Jack's Car Rental MDP")
    
    """
    mdp = JacksCarRental(
        max_cars=20,
        max_movement=5,
        average_hires=(3, 4),
        average_returns=(3, 2)
    )
    """
    mdp = JacksCarRental(
        max_cars=5,
        max_movement=3,
        average_hires=(3, 4),
        average_returns=(3, 2)
    )
    
    v = uniform_value_estimation(mdp)
    p = UniformPolicy(mdp, 0)

    print("Done initializing")

    iteration_delay = 1
    def on_iteration(k, v, p, v_new, p_new):
        plt.clf()
        mdp.generate_contour_figure(p_new)
        plt.title("Jack's Car Rental policy after {} iteration(s)".format(k))
        plt.pause(iteration_delay)


    fig = plt.figure()
    mdp.generate_contour_figure(p)
    plt.title("Jack's Car Rental policy after 0 iteration(s)")
    plt.show(block=False)
    plt.pause(iteration_delay)

    print("Applying policy iteration...")
    vstar, pstar = policy_iteration(
        mdp,
        v,
        p,
        on_iteration=on_iteration,
    )

    print("Done")

    mdp.generate_contour_figure(pstar)
    plt.title("Jack's Car Rental $\pi*$")
    plt.show()


if __name__ == "__main__":
    main()

"""
An implementation of the small gridworld MDP from David Silver's Reinforcement Learning
lecture series, lecture 3, p11
"""

import numpy as np
import matplotlib.pyplot as plt

# Get the MDP directory in our path
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mdp import GridWorld
from mdp import UniformRandomPolicy, uniform_value_estimation, evaluate_policy, policy_iteration


def main():
    """
    Excercises the GridWorld() class
    """
    small_gw = GridWorld.from_array(
        [
            [ 't',  '1',  '2',  '3'],
            [ '4',  '5',  '6',  '7'],
            [ '8',  '9', '10', '11'],
            ['12', '13', '14',  't'],
        ],
        lambda s: s == 't',
        action_set=GridWorld.ActionSetCompassFour,
        boundary_result="nothing",
        discount_factor=1,
        timestep_reward=-1,
        terminal_reward=0,
        wind_prob=0
    )
    print(small_gw)

    # Prepare initial estimates
    pi0 = UniformRandomPolicy(small_gw)
    v0 = uniform_value_estimation(small_gw)

    # Apply policy iteration
    vstar, pistar = policy_iteration(small_gw, v0, pi0, max_iterations=100)

    small_gw.generate_figure(
        value_function=vstar,
        policy=pistar,
        title=r"Small GridWorld - $\pi*$ and $V*$",
        subtitle="From David Silver's RL Lecture #3, p13"
    )
    plt.show()



if __name__ == "__main__":
    main()

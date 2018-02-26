"""
An implementation of the small gridworld MDP from David Silver's Reinforcement Learning
lecture series, lecture 3, p11
"""

import numpy as np

# Get the MDP directory in our path
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mdp import GridWorld
from mdp import UniformRandomPolicy, iterative_policy_evaluation


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
        terminal_reward=10,
        wind_prob=0
    )
    print(small_gw)
    meh_policy = UniformRandomPolicy(small_gw)

    v_pi = iterative_policy_evaluation(small_gw, meh_policy, max_iterations=100)
    print(v_pi)
    #print(GridWorld.as_grid(v_pi))


if __name__ == "__main__":
    main()

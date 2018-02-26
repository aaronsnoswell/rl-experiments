"""
An implementation of the small gridworld MDP from David Silver's Reinforcement Learning
lecture series, lecture 3, p11
"""

import numpy as np

# Get the MDP directory in our path
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mdp import GridWorld


def main():
    """
    Excercises the GridWorld() class
    """
    test = GridWorld.from_array(
        [
            [ 't',  '1',  '2',  '3'],
            [ '4',  '5',  '6',  '7'],
            [ '8',  '9', '10', '11'],
            ['12', '13', '14',  't'],
        ],
        lambda s: s == 't',
        action_set=GridWorld.ActionSetCompassFour,
        boundary_result="disallow",
        discount_factor=1,
        timestep_reward=-1,
        terminal_reward=10,
        wind_prob=0
    )
    print(test)


if __name__ == "__main__":
    main()

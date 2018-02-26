"""
An implementation of the small gridworld MDP from David Silver's Reinforcement Learning
lecture series, lecture 3, p11
"""

import numpy as np

# Get the MDP directory in our path
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mdp import GridWorld
from mdp import UniformRandomPolicy, GreedyPolicy, iterative_policy_evaluation


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


    def printable_policy(pol):
        """
        Helper function to get a display-friendly version of a policy
        """
        disp_pol = {}
        for key in pol:

            # Compute binary action vector
            action_bv = np.array(list(pol[key].values())) != 0

            if np.all(action_bv == [0, 0, 0, 0]):
                disp_pol[key] = ' ■'
            elif np.all(action_bv == [1, 1, 1, 1]):
                disp_pol[key] = ' +'
            elif np.all(action_bv == [1, 0, 0, 0]):
                disp_pol[key] = ' ↑'
            elif np.all(action_bv == [0, 1, 0, 0]):
                disp_pol[key] = ' →'
            elif np.all(action_bv == [0, 0, 1, 0]):
                disp_pol[key] = ' ↓'
            elif np.all(action_bv == [0, 0, 0, 1]):
                disp_pol[key] = ' ←'
            elif np.all(action_bv == [1, 1, 0, 0]):
                disp_pol[key] = '↑→'
            elif np.all(action_bv == [1, 0, 1, 0]):
                disp_pol[key] = ' ↕'
            elif np.all(action_bv == [1, 0, 0, 1]):
                disp_pol[key] = "↑←"
            elif np.all(action_bv == [0, 1, 1, 0]):
                disp_pol[key] = '↓→'
            elif np.all(action_bv == [0, 1, 0, 1]):
                disp_pol[key] = ' ↔'
            elif np.all(action_bv == [0, 0, 1, 1]):
                disp_pol[key] = '↓←'
            else:
                disp_pol[key] = '  '

        return disp_pol


    def on_iteration(k, v):
        gp = GreedyPolicy(small_gw, v)

        print(k)
        print(GridWorld.dict_as_grid(v))
        print(GridWorld.dict_as_grid(printable_policy(gp.policy_mapping)))
        
        input("Press any key to continue")
    

    meh_policy = UniformRandomPolicy(small_gw)
    v_pi = iterative_policy_evaluation(
        small_gw,
        meh_policy,
        max_iterations=100,
        #on_iteration=on_iteration
    )
    print(GridWorld.dict_as_grid(v_pi))
    print(
        GridWorld.dict_as_grid(
            printable_policy(
                GreedyPolicy(
                    small_gw,
                    v_pi
                ).policy_mapping
            )
        )
    )



if __name__ == "__main__":
    main()

"""
An implementation of the small gridworld MDP from David Silver's Reinforcement Learning
lecture series, lecture 3, p11
"""

import numpy as np
import matplotlib.pyplot as plt

# Get the MDP directory in our path
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mdp import *


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
    policy = UniformRandomPolicy(small_gw)
    value_function = small_gw.uniform_value_estimate()

    # Prepare plotting variables
    interim_figure_title = r"Small GridWorld with $\pi$ and $V$ for Policy Iteration k={}"
    final_figure_title = r"Small GridWorld with $\pi*$ and $V*$ after Policy Iteration"
    figure_subtitle = "From David Silver's RL Lecture #3, p13"


    def draw_figure(value_function, policy, title, subtitle):
        # Helper function to draw the figure
        
        plt.clf()
        small_gw.generate_figure(
            value_function=value_function,
            policy=policy,
            title=title,
            subtitle=subtitle
        )


    # Draw initial figure
    draw_figure(
        value_function,
        policy,
        interim_figure_title.format(0),
        figure_subtitle
    )
    plt.show(block=False)


    iteration_delay = 0.1
    def on_iteration(k, v, p, v_new, p_new):
        # Callback for each iteration
        
        draw_figure(
            v_new,
            p_new,
            interim_figure_title.format(k),
            figure_subtitle
        )
        plt.pause(iteration_delay)


    value_function, policy = Policy.policy_iteration(
        policy,
        value_function,
        on_iteration=on_iteration
    )

    print("Done")
    draw_figure(
        value_function,
        policy,
        final_figure_title,
        figure_subtitle
    )
    plt.show()



if __name__ == "__main__":
    main()

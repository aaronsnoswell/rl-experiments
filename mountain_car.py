"""
Provides an MDP class implementation for the OpenAI Gym continuous mountain
car problem
"""

import math
import numpy as np

from mdp.markov_decision_process import MarkovDecisionProcess
from mdp.policy import Policy, UniformRandomPolicy


class ContinuousMountainCar(MarkovDecisionProcess):
    """
    Defines an MDP for the MountainCar problem
    """

    def __init__(self, env, *, gamma=0.9, N=(120, 120)):
        """
        Constructor
        
        @param env - OpenAI gym environment object
            i.e. env == gym.make('MountainCarContinuous-v0')

        @param gamma - Discount factor
        @param N - Vector of discretisation sizes to use for the state space,
            |N| == 2
        """

        self.state_set = [
            (p, v) for v in np.linspace(
                -env.unwrapped.max_speed,
                env.unwrapped.max_speed,
                N[0]
            )
            for p in np.linspace(
                env.unwrapped.min_position,
                env.unwrapped.max_position,
                N[1]
            )
        ]

        # Temp reward vector for calculations
        # TODO ajs 24/Apr/2018 Read actual reward definition from env object
        R = [0 if s[0] < env.unwrapped.goal_position else 100 for s in self.state_set]

        # Define terminal state set
        self.terminal_state_set = [s for si, s in enumerate(self.state_set) if R[si] != 0]

        # Action space
        # NB: We only support a discrete action space, even though the gym
        # implementation is continuous
        self.action_set = [-1, 0, 1]


        # Transition function for the underlying gym environment
        def T(s, action):
            env.reset()
            env.unwrapped.state = s
            observation, reward, done, info = env.step([action])
            return observation


        def nearest_in_list(x, lst):
            """
            Helper function to find the nearest entry in lst to x
            """
            nearest_index = -1
            nearest_dist = math.inf
            for li, i in enumerate(lst):
                dist = np.linalg.norm(np.array(x) - np.array(i))
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_index = li
            return lst[nearest_index]


        # Estimate full transition tensor
        Tm = np.zeros(shape=[
                len(self.state_set),
                len(self.action_set),
                len(self.state_set)
            ],
            dtype=float
        )
        for s0i, s0 in enumerate(self.state_set):
            for ai, a in enumerate(self.action_set):

                # Step, performing the same action, until we change states
                s1curr = s0
                s1round = s0
                tol = 1e-6
                while s1round[0] == s0[0] and s1round[1] == s0[1]:

                    # Step with action a to get a new state
                    s1new = T(s1curr, a)


                    if np.linalg.norm(s1new - s1curr) < tol:
                        # If we're stationary, exit
                        s1curr = s1new
                        break

                    # Round to the nearest discretised state
                    s1curr = s1new
                    s1round = nearest_in_list(s1curr, self.state_set)

                # We either moved to a new state, or aren't moving at all
                s1i = self.state_set.index(s1round)
                Tm[s0i, ai, s1i] = 1

        # Set transition matrix
        self.transition_matrix = np.zeros(shape=[len(self.state_set) * len(self.action_set), len(self.state_set)])
        for s0i, s0 in enumerate(self.state_set):
            for ai, a in enumerate(self.action_set):
                self.transition_matrix[s0i * len(self.action_set) + ai, :] = Tm[s0i, ai, :]

        # Reward mapping
        self.reward_mapping = {s: {a: R[si] for a in self.action_set} for si, s in enumerate(self.state_set)}

        # In MountainCar, all actions are possible all the time
        self.possible_action_mapping = {s: self.action_set for s in self.state_set}

        # Set discount factor
        self.discount_factor = gamma


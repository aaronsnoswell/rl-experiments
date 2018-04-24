"""
Implementation of Trajectory-based Linear Programming IRL by Ng and Russell,
2000
(c) 2018 Aaron Snoswell
"""

import math
import copy
import numpy as np

from cvxopt import matrix, solvers


def tlp_irl(start_state, step, zeta, S_bounds, A, p=2.0, m=10,
        max_trajectory_length=5000, verbose=False):
    """
    Implements trajectory-based Linear Programming IRL by Ng and Russell, 2000
    
    @param start_state - The starting state for all monte-carlo sampled
        trajectories
    @param step - A function f(s, a) that returns a new state after applying
        action a from state s (and also returns a bool indicating if the MDP
        has terminated)
    @param zeta - A list of lists of state, action tuples. The expert's
        demonstrated trajectories provided to the algorithm
    @param S_bounds - List of tuples indicating the bounds of the state space
    @param A - The MDP action space
    """


    def state_indices(s, N):
        """
        Discretises the given state into it's appropriate indices, as a tuple
        """
        indices = []
        for state_index, state in enumerate(s):
            s_min, s_max = S_bounds[state_index]
            state_disc_index = round((state - s_min) / (s_max - s_min) * (N-1))
            state_disc_index = min(max(state_disc_index, 0), N-1)
            indices.append(int(state_disc_index))
        return tuple(indices)


    def randompolicy(N=20):
        """
        Generates a uniform random policy lookup tensor by discretising the
        state space

        @param N - The number of discretisation steps to use for each state
            dimension
        """
        r_policy = np.random.choice(A, [N] * len(S_bounds))
        return (lambda policy: lambda s: policy[state_indices(s, N)])(r_policy)


    def mc_trajectories(policy, step, m=m, max_trajectory_length=max_trajectory_length):
        """
        Sample some monte-carlo trajectories from the given policy tensor

        @param policy - A function f(s) that returns the next action as a
            function of our current state
        @param step - A function f(s, a) that returns a new state
            after applying action a from state s (and also returns a bool
            indicating if the MDP has terminated)
        """
        trajectories = []
        for i in range(m):
            print(i)

            state = start_state
            rollout = []

            while len(rollout) < max_trajectory_length:
                action = policy(state)
                rollout.append((state, action))

                state, done = step(state, action)
                if done: break
            trajectories.append(rollout)

        return trajectories


    rp = randompolicy()
    trajs = mc_trajectories(rp, step)

    pass



if __name__ == "__main__":

    """
    # Collect a single trajectory from the human user
    from gym_mountaincar import run_episode, manual_policy
    _, _, zeta = run_episode(manual_policy)
    """
    zeta = []

    # Create tmp MC object so we can read properties of the MDP
    import gym
    env = gym.make('MountainCarContinuous-v0')

    # Define our step function that allows us to simulate forwards through the world dynamics
    def step(state, action):
        env.reset()
        env.unwrapped.state = state
        state, reward, done, status = env.step([action])
        return state, done


    tlp_irl(
        np.array([0, 0]),
        step,
        zeta,
        [
            (env.unwrapped.min_position, env.unwrapped.max_position),
            (-env.unwrapped.max_speed, env.unwrapped.max_speed)
        ],
        [-1, 0, 1]
    )



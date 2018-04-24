"""
Implementation of Trajectory-based Linear Programming IRL by Ng and Russell,
2000
(c) 2018 Aaron Snoswell
"""

import math
import copy
import numpy as np

from cvxopt import matrix, solvers


def tlp_irl(start_state, step, zeta, S_bounds, A, phi, gamma, *, p=2.0, m=10,
        max_trajectory_length=5000, tol=1e-6, verbose=False):
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
    @param phi - A vector of d basis functions phi_i(s) mapping from S to real
        numbers
    @param gamma - Discount factor for future rewards
    """

    d = len(phi)


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


    def random_policy(N=20):
        """
        Generates a uniform random policy lookup tensor by discretising the
        state space

        @param N - The number of discretisation steps to use for each state
            dimension
        """
        r_policy = np.random.choice(A, [N] * len(S_bounds))
        return (lambda policy: lambda s: policy[state_indices(s, N)])(r_policy)


    def mc_trajectories(policy, step, phi, m=m, max_trajectory_length=max_trajectory_length):
        """
        Sample some monte-carlo trajectories from the given policy tensor

        @param policy - A function f(s) that returns the next action as a
            function of our current state
        @param step - A function f(s, a) that returns a new state
            after applying action a from state s (and also returns a bool
            indicating if the MDP has terminated)
        @param phi - A vector of d reward function basis functions mapping
            from S to real numbers
        """
        trajectories = []
        for i in range(m):

            state = start_state
            rollout = []

            while len(rollout) < max_trajectory_length:
                action = policy(state)
                rollout.append((state, action))

                state, done = step(state, action)
                if done: break
            trajectories.append(rollout)

        return trajectories


    def emperical_trajectory_value(trajectory):
        """
        Computes the vector of discounted future basis function values for the
        given trajectory. The inner product of this vector and an alpha vector
        gives the emperical value of zeta under the reward function defined by
        alpha
        """
        value = np.zeros(shape=d)
        for i in range(d):
            phi_i = phi[i]
            for j in range(len(trajectory)):
                value[i] += gamma ** j * phi_i(trajectory[j][0])
        return value


    def emperical_policy_value(zeta):
        """
        Computes the vector of mean discounted future basis function values
        for the given set of trajectories representing a single policy.
        """
        value_vector = np.zeros(shape=d)
        for trajectory in zeta:
            value_vector += emperical_trajectory_value(trajectory)
        value_vector /= len(zeta)


    # Initialize alpha vector
    alpha = np.random.uniform(size=d) * 2 - 1

    # Compute mean expert trajectory value vector
    expert_value_vector = emperical_policy_value(zeta)

    # Initialize the non-expert policy set with a random policy
    non_expert_policy_set = [random_policy()]
    non_expert_policy_value_vectors = np.array([
        emperical_policy_value(
            mc_trajectories(
                non_expert_policy_set[0],
                step,
                phi
            )
        )
    ])

    # Loop until reward convergence
    while True:

        # Formulate the LP problem such that the expert's value function is
        # greater than all non-expert value functions

        # Solve the LP problem

        # Find a new greedy policy based on the new alpha vector

        # Add this policy to the list of non-expert policies
        
        # If alpha_i's have converged, break
        break


    return alpha



if __name__ == "__main__":

    # Collect a single trajectory from the human user
    from gym_mountaincar import run_episode, manual_policy
    _, _, zeta = run_episode(manual_policy)

    # Create tmp MC object so we can read properties of the MDP
    import gym
    env = gym.make('MountainCarContinuous-v0')


    # Define our step function that allows us to simulate forwards through the world dynamics
    def step(state, action):
        env.reset()
        env.unwrapped.state = state
        state, reward, done, status = env.step([action])
        return state, done


    def normal(mu, sigma, x):
        """
        1D Normal function
        """
        return math.exp(-1 * ((x - mu) ** 2) / (2 * sigma ** 2)) / \
            math.sqrt(2 * math.pi * sigma ** 2)


    # Build a set of basis functions
    d = 4
    sigma = 0.1
    min_pos = env.unwrapped.min_position
    max_pos = env.unwrapped.max_position
    delta = (max_pos - min_pos) / d
    phi = [
        (lambda mu: lambda s: normal(mu, sigma, s[0]))(p) for p in np.arange(
            min_pos + delta/2,
            max_pos + delta/2,
            delta
        )
    ]

    # Perform trajectory based IRL
    tlp_irl(
        np.array([0, 0]),
        step,
        [zeta],
        [
            (env.unwrapped.min_position, env.unwrapped.max_position),
            (-env.unwrapped.max_speed, env.unwrapped.max_speed)
        ],
        [-1, 0, 1],
        phi,
        0.9
    )



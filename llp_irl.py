"""
Implementation of Linear Programming IRL for large state spaces by Ng and
Russell, 2000
(c) 2018 Aaron Snoswell
"""

import math
import copy
import numpy as np

from cvxopt import matrix, solvers


def llp_irl(sf, M, k, T, phi, *, N=1000, p=2.0, verbose=False):
    """
    Implements Linear Programming IRL for large state spaces by NG and
        Russell, 2000

    See https://thinkingwires.com/posts/2018-02-13-irl-tutorial-1.html for a
        good reference.
    
    @param sf - A state factory function that takes no arguments and returns
        an i.i.d. sample from the MDP state space
    @param M - The number of sub-samples to draw from the state space |S_0|
        when estimating the expert's reward function
    @param k - The number of actions |A|
    @param T - A sampling transition function T(s, ai) -> s' encoding a
        stationary deterministic policy. The structure of T must be that the
        0th action T(:, 0) corresponds to a sample from the expert policy, and
        T(:, i), i != 0 corresponds to a sample from the ith non-expert action
        at each state, for some arbitrary but consistent ordering of states
    @param phi - A vector of d basis functions phi_i(s) mapping from S to real
        numbers
    @param N - Number of transition samples to use when computing expectations
        over the Value basis functions
    @param p - Penalty function coefficient. Ng and Russell find p=2 is robust
        Must be >= 1
    @param verbose - If true, progress information will be shown

    @return A vector of d 'alpha' coefficients for the basis functions phi(S)
        that allows rewards to be computed for a state via the inner product
        alpha_i · phi
    @return A result object from the optimiser
    """

    # Measure number of basis functions
    d = len(phi)

    # Enforce valid penalty function coefficient
    assert m >= 1, \
        "Penalty function coefficient must be >= 1, was {}".format(m)

    def compute_value_expectation_tensor(sf, M, k, T, phi, N, verbose):
        """
        Computes the value expectation tensor VE

        This is an array of shape (d, k-1, M) where VE[:, i, j] is a vector of
        coefficients that, when multiplied with the alpha vector give the
        expected difference in value between the expert policy action and the
        ith action from state s_j

        @param sf - A state factory function that takes no arguments and returns
            an i.i.d. sample from the MDP state space
        @param M - The number of sub-samples to draw from the state space |S_0|
            when estimating the expert's reward function
        @param k - The number of actions |A|
        @param T - A sampling transition function T(s, ai) -> s' encoding a
            stationary deterministic policy. The structure of T must be that the
            0th action T(:, 0) corresponds to a sample from the expert policy, and
            T(:, i), i != 0 corresponds to a sample from the ith non-expert action
            at each state, for some arbitrary but consistent ordering of states
        @param phi - A vector of d basis functions phi_i(s) mapping from S to real
            numbers
        @param N - Number of transition samples to use when computing expectations
            over the value basis functions
        @param verbose - If true, progress information will be shown

        @return The value expectation tensor VE. A numpy array of shape
            (d, k-1, M)
        """


        def expectation(fn, sf, N):
            """
            Helper function to estimate an expectation over some function fn(sf())

            @param fn - A function of a single variable that the expectation will
                be computed over
            @param sf - A state factory function - takes no variables and returns
                an i.i.d. sample from the state space
            @param N - The number of draws to use when estimating the expectation

            @return An estimate of the expectation E[fn(sf())]
            """
            state = sf()
            return sum([fn(sf()) for n in range(N)]) / N


        # Measure number of basis functions
        d = len(phi)

        # Prepare tensor
        VE_tensor = np.zeros(shape=(d, k-1, M))

        # Draw M initial states from the state space
        for j in range(M):
            if verbose: print("{} / {}".format(j, M))
            s_j = sf()

            # Compute E[phi(s')] where s' is drawn from the expert policy
            expert_basis_expectations = np.array([
                expectation(phi[di], lambda: T(s_j, 0), N) for di in range(d)
            ])

            # Loop over k-1 non-expert actions
            for i in range(1, k):

                # Compute E[phi(s')] where s' is drawn from the ith non-expert action
                ith_non_expert_basis_expectations = np.array([
                    expectation(phi[di], lambda: T(s_j, i), N) for di in range(d)
                ])

                # Compute and store the expectation difference for this initial
                # state
                test = expert_basis_expectations - \
                    ith_non_expert_basis_expectations

                VE_tensor[:, i-1, j] = test

        return VE_tensor


    # Precompute the value expectation tensor VE
    # This is an array of shape (d, k-1, M) where VE[:, i, j] is a vector of
    # coefficients that, when multiplied with the alpha vector give the
    # expected difference in value between the expert policy action and the
    # ith action from state s_j
    if verbose: print("Computing expectations...")
    VE_tensor = compute_value_expectation_tensor(sf, M, k, T, phi, N, verbose)

    # Formulate the linear programming problem constraints
    # NB: The general form for adding a constraint looks like this
    # c, A_ub, b_ub = f(c, A_ub, b_ub)
    if verbose: print("Composing LP problem...")


    def add_costly_single_step_constraints(c, A_ub, b_ub):
        """
        Augments the objective and adds constraints to implement the Linear
        Programming IRL method for large state spaces

        This will add M extra variables and 2M*(k-1) constraints

        NB: Assumes the true optimisation variables are first in the c vector
        """

        # Step 1: Add the extra optimisation variables for each min{} operator
        # (one per sampled state)
        c = np.hstack([np.zeros(shape=(1, d)), np.ones(shape=(1, M))])
        A_ub = np.hstack([A_ub, np.zeros(shape=(A_ub.shape[0], M))])

        # Step 2: Add the constraints

        # Loop for each of the starting sampled states s_j
        for j in range(M):

            # Loop over the k-1 non-expert actions
            for i in range(1, k):

                # Add two constraints, one for each half of the penalty
                # function p(x)
                constraint_row = np.hstack([VE_tensor[:, i-1, j], \
                    np.zeros(shape=M)])
                constraint_row[d + j] = -1
                A_ub = np.vstack((A_ub, constraint_row))
                b_ub = np.vstack((b_ub, 0))

                constraint_row = np.hstack([p * VE_tensor[:, i-1, j], \
                    np.zeros(shape=M)])
                constraint_row[d + j] = -1
                A_ub = np.vstack((A_ub, constraint_row))
                b_ub = np.vstack((b_ub, 0))

        return c, A_ub, b_ub


    def add_alpha_size_constraints(c, A_ub, b_ub):
        """
        Add constraints for a maximum |alpha| value of 1
        This will add 2 * d extra constraints

        NB: Assumes the true optimisation variables are first in the c vector
        """
        for i in range(d):
            constraint_row = [0] * A_ub.shape[1]
            constraint_row[i] = 1
            A_ub = np.vstack((A_ub, constraint_row))
            b_ub = np.vstack((b_ub, 1))

            constraint_row = [0] * A_ub.shape[1]
            constraint_row[i] = -1
            A_ub = np.vstack((A_ub, constraint_row))
            b_ub = np.vstack((b_ub, 1))
        return c, A_ub, b_ub


    # Prepare LP constraint matrices
    c = np.zeros(shape=[1, d], dtype=float)
    A_ub = np.zeros(shape=[0, d], dtype=float)
    b_ub = np.zeros(shape=[0, 1])

    # Compose LP optimisation problem
    c, A_ub, b_ub = add_costly_single_step_constraints(c, A_ub, b_ub)
    c, A_ub, b_ub = add_alpha_size_constraints(c, A_ub, b_ub)

    # Show the LP system prior to solving
    #print(c[0, :])
    #print(A_ub)
    #print(b_ub[:, 0])
    if verbose:
        print("Number of optimsation variables: {}".format(c.shape[1]))
        print("Number of constraints: {}".format(A_ub.shape[0]))

    # Solve for a solution
    if verbose: print("Solving LP problem...")
    
    # NB: cvxopt.solvers.lp expects a 1d c vector
    res = solvers.lp(matrix(c[0, :]), matrix(A_ub), matrix(b_ub))

    # Extract the true optimisation variables
    alpha_vector = res['x'][0:d].T

    return alpha_vector, res



if __name__ == "__main__":

    # Construct an IRL problem from the MountainCar benchmark
    print("Preparing MountainCar problem")
    import gym
    env = gym.make('MountainCarContinuous-v0')
    env._max_episode_steps = 999

    # Lambda that returns i.i.d. samples from state space
    sf = lambda: [
            np.random.uniform(env.unwrapped.min_position, \
                env.unwrapped.max_position),
            np.random.uniform(-env.unwrapped.max_speed, \
                env.unwrapped.max_speed)
        ]

    # Number of states to use for reward function estimation
    M = 100

    # There are three possible actions
    action_set = [0, 1, 2]
    k = len(action_set)

    # A simple 'expert' policy that solves the mountain car problem
    simple_policy = lambda s: 0 if (s[1] - 0.003) < 0 else 2

    # Transition function
    def T(s, action):
        env.reset()
        env.unwrapped.state = s
        observation, reward, done, info = env.step([action])
        return observation

    # Transition function
    def T_nonexpert(s, action_index):
        # Find the expert action at our current state
        expert_action = simple_policy(s)
        non_expert_action_set = copy.copy(action_set)
        non_expert_action_set.remove(expert_action)

        possible_action_set = [expert_action] +  non_expert_action_set
        return T(s, possible_action_set[action_index])


    def normal(mu, sigma, x):
        """
        1D Normal function
        """

        return math.exp(-1 * ((x - mu) ** 2) / (2 * sigma ** 2)) / \
            math.sqrt(2 * math.pi * sigma ** 2)


    # Build basis function set of size |d|
    d = 3
    simga = 0.5
    min_pos = env.unwrapped.min_position
    max_pos = env.unwrapped.max_position
    step = (max_pos - min_pos) / d
    basis_function_positions = np.arange(
        min_pos + step/2, max_pos + step/2, step
    )

    phi = [
        lambda s: normal(mu, simga, s[0]) for mu in basis_function_positions
    ]

    # Solve the MDP using state, action discretisation for funtion
    # approximation
    print("Preparing discrete approximation")
    S = [
        (p, v) for v in np.linspace(
            -env.unwrapped.max_speed,
            env.unwrapped.max_speed,
            20
        )
        for p in np.linspace(
            env.unwrapped.min_position,
            env.unwrapped.max_position,
            5
        )
    ]
    A = [0, 1, 2]
    R = [0 if s[0] < env.unwrapped.goal_position else 100 for s in S]
    gamma = 0.9

    def nearest_in_list(x, lst):
        nearest_index = -1
        nearest_dist = math.inf
        for li, i in enumerate(lst):
            dist = np.linalg.norm(np.array(x) - np.array(i))
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_index = li
        return lst[nearest_index]

    # Estimate full transition matrix
    Tm = np.zeros(shape=[len(S), len(A), len(S)], dtype=float)
    for s0i, s0 in enumerate(S):
        for ai, a in enumerate(A):

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
                s1round = nearest_in_list(s1curr, S)

            # We either moved to a new state, or aren't moving at all
            s1i = S.index(s1round)
            Tm[s0i, ai, s1i] = 1

    Tm_sa = np.zeros(shape=[len(S) * len(A), len(S)])
    for s0i, s0 in enumerate(S):
        for ai, a in enumerate(A):
            Tm_sa[s0i * len(A) + ai, :] = Tm[s0i, ai, :]

    from mdp.markov_decision_process import MarkovDecisionProcess
    from mdp.policy import Policy, UniformRandomPolicy

    print("Preparing MDP")
    class MCMDP(MarkovDecisionProcess):
        """
        Define an MDP for the MountainCar problem
        """

        def __init__(self):
            self.state_set = S
            self.terminal_state_set = [s for si, s in enumerate(S) if R[si] != 0]
            self.action_set = A
            self.transition_matrix = Tm_sa
            self.reward_mapping = {s: {a: R[si] for a in A} for si, s in enumerate(S)}
            self.possible_action_mapping = {s: A for s in S}
            self.discount_factor = 0.9


    # Compute optimal policy via PI
    print("Solving for optimal policy")
    mc_mdp = MCMDP()
    v_star, p_star = Policy.policy_iteration(
        UniformRandomPolicy(mc_mdp),
        {s: 1/len(S) for s in S}
    )

    # Try running the discovered optimal policy on the task
    from gym_mountaincar import run_episode

    def p_fn(observation, env, key_handler):
        # Discretise observation
        s_disc = nearest_in_list(observation, S)
        return p_star.get_action(s_disc)

    run_episode(p_fn, continuous=True)


    """
    # Run IRL
    alpha_vector, res = llp_irl(sf, M, k, T_nonexpert, phi, verbose=True)


    # Compose reward function
    R = lambda s: np.dot(alpha_vector, [phi[i](s) for i in range(len(phi))])[0]

    print(alpha_vector)

    for sx in np.linspace(env.unwrapped.min_position, env.unwrapped.max_position, 100):
        print("{}, {}".format(sx, R([sx, 0])))
    """



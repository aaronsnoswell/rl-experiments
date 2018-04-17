"""
Implementation of Linear Programming IRL by Ng and Abbeel, 2000
(c) 2018 Aaron Snoswell
"""

import math
import copy
import numpy as np


def build_sorted_transition_matrix(S, A, T, pi):
    """
    Given a vector of states S, a vector of actions A, a transition matrix
    T(s-a, s') and a policy dictionary, builds a sorted transition matrix
    T(s, a, s'), where the 0th action T(:, 0, :) corresponds to the expert
    policy, and the ith action T(:, i, :), i!=0 corresponds to the ith non
    -expert action at each state
    """

    # Build the compact form Transition matrix
    n = len(S)
    k = len(A)

    # Helper function to get a transition probability
    si = lambda s: S.tolist().index(s)
    ai = lambda a: A.tolist().index(a)
    trans = lambda s1, a, s2: T[si(s1) * k + ai(a), si(s2)]

    Tfull = np.zeros(shape=[n, k, n])
    for s_from in S:
        # Build vector of actions, sorted with the expert one first
        expert_action = pi[s_from]
        sorted_actions = np.append([expert_action], np.delete(A, ai(expert_action)))
        for a in sorted_actions:
            for s_to in S:
                Tfull[si(s_from), ai(a), si(s_to)] = trans(s_from, a, s_to)

    return Tfull


def lp_irl(T, gamma, l1, *, Rmax=1.0, method="cvxopt"):
    """
    Implements Linear Programming IRL by NG and Abbeel, 2000

    Given a transition matrix T(s, a, s') encoding a stationary, deterministic
    policy and a discount factor gamma finds a reward vector R(s) for which
    the policy is optimal.

    This method uses the Linear Programming IRL algorithm by Ng and Abbeel,
    2000 (http://ai.stanford.edu/~ang/papers/icml00-irl.pdf). See
    https://www.inf.ed.ac.uk/teaching/courses/rl/slides17/8_IRL.pdf for a more
    accessible overview.

    @param T - A sorted transition matrix T(s, a, s') encoding a stationary
        deterministic policy. The structure of T must be that the 0th action
        T[:, 0, :] corresponds to the expert policy, and T[:, i, :], i != 0
        corresponds to the ith non-expert action at each state
    @param gamma - The expert's discount factor
    @param l1 - L1 regularization weight for LP optimisation objective
        function
    @param Rmax - Maximum reward value
    @param method - LP programming method. One of "cvxopt", "scipy-simplex" or
        "scipy-interior-point"

    @return A reward vector for which the given policy is optimal
    @return A result object from the optimiser

    TODO: Adjust L1 norm constraint generation to allow negative rewards in
    the final vector. 
    """

    # Measure size of state and action sets
    n = T.shape[0]
    k = T.shape[1]

    # Compute the discounted transition matrix term
    T_disc_inv = np.linalg.inv(np.identity(n) - gamma * T[:, 0, :])

    # Formulate the linear programming problem constraints
    # NB: The general form for adding a constraint looks like this
    # c, A_ub, b_ub = f(c, A_ub, b_ub)

    # Prepare LP constraint matrices
    c = np.zeros(shape=[1, n], dtype=float)
    A_ub = np.zeros(shape=[0, n], dtype=float)
    b_ub = np.zeros(shape=[0, 1])


    def add_optimal_policy_constraints(c, A_ub, b_ub):
        """
        Add constraints to ensure the expert policy is optimal
        This will add (k-1) * n extra constraints
        """
        for i in range(k - 1):
            constraint_rows = -1 * (T[:, 0, :] - T[:, i, :]) @ T_disc_inv
            A_ub = np.vstack((A_ub, constraint_rows))
            b_ub = np.vstack((b_ub, np.zeros(shape=[constraint_rows.shape[0], 1])))
        return c, A_ub, b_ub


    def add_costly_single_step_constraints(c, A_ub, b_ub):
        """
        Augment the optimisation objective to add the costly-single-step
        degeneracy heuristic
        This will add n extra optimisation variables and (k-1) * n extra
        constraints
        NB: Assumes the true optimisation variables are first in the objective
        function
        """

        # Expand the c vector add new terms for the min{} operator
        c = np.hstack((c, -1 * np.ones(shape=[1, n])))
        css_offset = c.shape[1] - n
        A_ub = np.hstack((A_ub, np.zeros(shape=[A_ub.shape[0], n])))

        # Add min{} operator constraints
        for i in range(k - 1):
            # Generate the costly single step constraint terms
            constraint_rows = -1 * (T[:, 0, :] - T[:, i, :]) @ T_disc_inv

            # constraint_rows is nxn - we need to add the min{} terms though
            min_operator_entries = np.identity(n)
            
            # And we have to make sure we put the min{} operator entries in
            # the correct place in the A_ub matrix
            num_padding_cols = css_offset - n
            padding_entries = np.zeros(shape=[constraint_rows.shape[0], num_padding_cols])
            constraint_rows = np.hstack((constraint_rows, padding_entries, min_operator_entries))

            # Finally, add the new constraints
            A_ub = np.vstack((A_ub, constraint_rows))
            b_ub = np.vstack((b_ub, np.zeros(shape=[constraint_rows.shape[0], 1])))
        
        return c, A_ub, b_ub


    def add_l1norm_constraints(c, A_ub, b_ub, l1):
        """
        Augment the optimisation objective to add an l1 norm regularisation
        term z += l1 * ||R||_1
        This will add n extra optimisation variables and 2n extra constraints
        NB: Assumes the true optimisation variables are first in the objective
        function
        """

        # We add an extra variable for each each true optimisation variable
        c = np.hstack((c, l1 * np.ones(shape=[1, n])))
        l1_offset = c.shape[1] - n

        # Don't forget to resize the A_ub matrix to match
        A_ub = np.hstack((A_ub, np.zeros(shape=[A_ub.shape[0], n])))

        # Now we add 2 new constraints for each true optimisation variable to
        # enforce the absolute value terms in the l1 norm
        for i in range(n):

            # An absolute value |x1| can be enforced via constraints
            # -x1 <= 0             (i.e., x1 must be positive or 0)
            #  x1 + -xe1 <= 0
            # Where xe1 is the replacement for |x1| in the objective
            #
            # TODO ajs 04/Apr/2018 This enforces that R must be positive or 0,
            # but I was under the impression that it was also possible to
            # enforce an abs operator without this requirement - e.g. see
            # http://lpsolve.sourceforge.net/5.1/absolute.htm
            constraint_row_1 = [0] * A_ub.shape[1]
            constraint_row_1[i] = -1
            A_ub = np.vstack((A_ub, constraint_row_1))
            b_ub = np.vstack((b_ub, [[0]]))

            constraint_row_2 = [0] * A_ub.shape[1]
            constraint_row_2[i] = 1
            constraint_row_2[l1_offset + i] = -1
            A_ub = np.vstack((A_ub, constraint_row_2))
            b_ub = np.vstack((b_ub, [[0]]))

        return c, A_ub, b_ub


    def add_rmax_constraints(c, A_ub, b_ub, Rmax):
        """
        Add constraints for a maximum R value Rmax
        This will add n extra constraints
        """
        for i in range(n):
            constraint_row = [0] * A_ub.shape[1]
            constraint_row[i] = 1
            A_ub = np.vstack((A_ub, constraint_row))
            b_ub = np.vstack((b_ub, Rmax))
        return c, A_ub, b_ub

    
    # Compose LP optimisation problem
    c, A_ub, b_ub = add_optimal_policy_constraints(c, A_ub, b_ub)
    c, A_ub, b_ub = add_costly_single_step_constraints(c, A_ub, b_ub)
    c, A_ub, b_ub = add_rmax_constraints(c, A_ub, b_ub, Rmax)
    c, A_ub, b_ub = add_l1norm_constraints(c, A_ub, b_ub, l1)

    # Show the LP system prior to solving
    #print(c[0, :])
    #print(A_ub)
    #print(b_ub[:, 0])

    # Solve for a solution
    res = None
    if method == "scipy-simplex":
        # NB: scipy.optimize.linprog expects a 1d c vector
        from scipy.optimize import linprog
        res = linprog(c[0, :], A_ub=A_ub, b_ub=b_ub[:, 0], method="simplex")

    elif method == "scipy-interior-point":
        # NB: scipy.optimize.linprog expects a 1d c vector
        from scipy.optimize import linprog
        res = linprog(c[0, :], A_ub=A_ub, b_ub=b_ub[:, 0],  method="interior-point")

    elif method == "cvxopt":
        # NB: cvxopt.solvers.lp expects a 1d c vector
        from cvxopt import matrix, solvers
        res = solvers.lp(matrix(c[0, :]), matrix(A_ub), matrix(b_ub))

    else:

        raise Exception("Unkown LP method type: {}".format(method))
        return None


    def normalize(vals):
        """
        normalize to (0, max_val)
        input:
        vals: 1d array
        """
        min_val = np.min(vals)
        max_val = np.max(vals)
        return (vals - min_val) / (max_val - min_val)

    
    # Extract the true optimisation variables and re-scale
    rewards = Rmax * normalize(res['x'][0:n])

    return rewards, res


def llp_irl(S0, k, T, phi, *, m=2.0, Rmax=1.0, method="cvxopt"):
    """
    Implements Linear Programming IRL for large state spaces by NG and Abbeel,
        2000

    See https://thinkingwires.com/posts/2018-02-13-irl-tutorial-1.html for a
        good reference.
    
    @param S0 - A sampled sub-set of the full state-space S
    @param k - The number of actions |A|
    @param T - A sampling transition function T(s, ai) -> s' encoding a
        stationary deterministic policy. The structure of T must be that the
        0th action T(:, 0) corresponds to a sample from the expert policy, and
        T(:, i), i != 0 corresponds to a sample from the ith non-expert action
        at each state, for some arbitrary but consistent ordering of states
    @param phi - A vector of d basis functions phi_i(s) mapping from S to the
        reals
    @param m - Penalty function coefficient. Ng and Abbeel find m=2 is robust
    @param Rmax - Maximum reward value
    @param method - LP programming method. One of "cvxopt", "scipy-simplex" or
        "scipy-interior-point"

    NB: method == scipy-interior-point depends on scikit-learn>=0.19.1

    @return A vector of d coefficients for the basis functions phi(S)
        that allows rewards to be computed for a state via the inner product
        alpha · phi
    @return A result object from the optimiser
    """

    # Measure number of sampled states and number of basis functions
    N = len(S0)
    d = len(phi)

    # Lambda for the penalty function
    penalty = lambda x: x if x >= 0 else m*x

    # Compute E[phi(expert_s') - phi(non_expert_s')] for all non-expert
    # policies. The phi_tensor[:, i] is the vector
    # (phi(s_expert') - phi(s_non_expert_i')) / N, where N is the number of
    # sampled states the expectation is computed over
    phi_tensor = np.zeros(shape=(d, k-1))
    for s in S0:
        expert_new_state = T(s, 0)

        for non_expert_action in range(k - 1):
            non_expert_new_state = T(s, non_expert_action + 1)

            phi_tensor[:, non_expert_action] += [phi_i(expert_new_state) - phi_i(non_expert_new_state) for phi_i in phi]
        phi_tensor /= N


    # Formulate the linear programming problem constraints
    # NB: The general form for adding a constraint looks like this
    # c, A_ub, b_ub, A_eq, b_eq = f(c, A_ub, b_ub, A_eq, b_eq)


    def add_costly_single_step_constraints(c, A_ub, b_ub, A_eq, b_eq):
        """
        Augment the optimisation objective to add the costly-single-step
        degeneracy heuristic
        This will add n extra optimisation variables and (k-1) * n extra
        constraints
        """

        # Extend the optimisation function, adding extra optimisation
        # variables for every entry in the min{} operator
        
        #c_size = c.shape[1]
        #for non_expert_action in range(k-1):
        #    c = np.hstack([c, [phi_tensor[:, non_expert_action].T]])
        #    A_ub = np.hstack([A_ub, np.zeros(shape=[A_ub.shape[0], k])])
        #    A_eq = np.hstack([A_eq, np.zeros(shape=[A_eq.shape[0], k])])

        # Add constraints to enforce the min{} operator
        # for ki in range(k-1):
        #     constraint_row = np.zeros(shape=[A_ub.shape[1]])
        #     constraint_row[min_offset] = 1
        #     constraint_row[min_offset + 1 + ki] = -1
        #     A_ub = np.vstack([A_ub, constraint_row])
        #     b_ub = np.vstack([b_ub, 0])

        # Add 

        return c, A_ub, b_ub, A_eq, b_eq



    def add_penalty_function_constraints(c, A_ub, b_ub, A_eq, b_eq, m=m):
        """
        Augments the objective function and adds constraints to account for
        a penalty function p(x) = x if x > 0 else m*x
        
        This will add |c| optimisation variables and |c| * 3 constraints

        NB: This function must be called after any LP constraints for it's
        arguments have been applied

        NB: Assumes the true optimisation variables are first in the c vector
        """

        # The penalty function is piecewise linear, with two pieces
        # Therefore we double the objective function terms, and apply the
        # appropriate constraint to each piece
        c_size = c.shape[1]
        c = np.hstack([c, m * c])
        A_ub = np.hstack((A_ub, np.zeros(shape=[A_ub.shape[0], c_size])))
        A_eq = np.hstack([A_eq, np.zeros(shape=[A_eq.shape[0], c_size])])

        # Add the constraints for the >= 0 part of the penalty function
        for i in range(d):
            constraint_row = np.zeros(shape=[A_ub.shape[1]])
            constraint_row[i] = -1
            A_ub = np.vstack([A_ub, constraint_row])
            b_ub = np.vstack([b_ub, 0])

        # Add the constraints for the < 0 part of the penalty function
        for i in range(d):
            constraint_row = np.zeros(shape=[A_ub.shape[1]])
            constraint_row[c_size + i] = 1
            A_ub = np.vstack([A_ub, constraint_row])
            b_ub = np.vstack([b_ub, 0])

        # Enforce that the alpha_i's must be equal on each side of the penalty
        # function
        for i in range(d):
            constraint_row = np.zeros(shape=[A_eq.shape[1]])
            constraint_row[i] = 1
            constraint_row[c_size + i] = -1
            A_eq = np.vstack([A_eq, constraint_row])
            b_eq = np.vstack([b_eq, 0])

        return c, A_ub, b_ub, A_eq, b_eq


    def add_alpha_size_constraints(c, A_ub, b_ub, A_eq, b_eq):
        """
        Add constraints for a maximum alpha value of 1
        This will add d extra constraints

        NB: Assumes the true optimisation variables are first in the c vector
        """
        for i in range(d):
            constraint_row = [0] * A_ub.shape[1]
            constraint_row[i] = 1
            A_ub = np.vstack((A_ub, constraint_row))
            b_ub = np.vstack((b_ub, 1))
        return c, A_ub, b_ub, A_eq, b_eq


    # Prepare LP constraint matrices
    c = np.zeros(shape=[1, d], dtype=float)
    A_ub = np.zeros(shape=[0, d], dtype=float)
    b_ub = np.zeros(shape=[0, 1])
    A_eq = np.zeros(shape=[0, d], dtype=float)
    b_eq = np.zeros(shape=[0, 1])

    # Compose LP optimisation problem
    c, A_ub, b_ub, A_eq, b_eq = add_costly_single_step_constraints(c, A_ub, b_ub, A_eq, b_eq)
    #c, A_ub, b_ub, A_eq, b_eq = add_penalty_function_constraints(c, A_ub, b_ub, A_eq, b_eq)
    #c, A_ub, b_ub, A_eq, b_eq = add_alpha_size_constraints(c, A_ub, b_ub, A_eq, b_eq)

    print(c)
    print(A_ub)
    print(b_ub)
    print(A_eq)
    print(b_eq)






if __name__ == "__main__":

    """
    # Sample problems for lp_irl
    # An n=3 problem
    T = build_sorted_transition_matrix(
        np.array(["s0", "s1", "s2"]),
        np.array(["b", "o"]),
        np.array([[0,    0.4, 0.6 ],
                  [0,    0,   1   ],
                  [0,    0,   1   ],
                  [0,    0,   1   ],
                  [1,    0,   0   ],
                  [1,    0,   0   ]]),
        {
            "s0": "b",
            "s1": "o",
            "s2": "o"
        }
    )

    # Try a smaller (n=2) problem
    T = build_sorted_transition_matrix(
        np.array(["s0", "s1"]),
        np.array(["b", "o"]),
        np.array([[0.4, 0.6],
                  [0.9, 0.1],
                  [1,   0],
                  [1,   0]]),
        {
          "s0": "b",
          "s1": "o"
        }
    )

    # Try LP IRL
    print(T)
    rewards, _ = lp_irl(T, 0.9, l1=10)
    print(rewards)
    """

    # Construct an IRL problem from the MountainCar benchmark
    import gym
    env = gym.make('MountainCar-v0')

    # Sample to build state set
    S0 = []
    N = 1000
    for ni in range(N):
        S0.append(np.random.uniform(
            low=env.unwrapped.low,
            high=env.unwrapped.high
        ))

    # There are three possible actions
    action_set = [0, 1, 2]
    k = len(action_set)

    # A simple 'expert' policy that solves the mountain car problem
    simple_policy = lambda s: 0 if (s[1] - 0.003) < 0 else 2

    # Transition function
    def T(s, action_index):
        env.reset()
        env.unwrapped.state = s

        # Find the expert action at our current state
        expert_action = simple_policy(s)
        non_expert_action_set = copy.copy(action_set)
        non_expert_action_set.remove(expert_action)

        possible_action_set = [expert_action] +  non_expert_action_set

        observation, reward, done, info = env.step(possible_action_set[action_index])
        return observation


    def normal(mu, sigma, x):
        """
        1D Normal function
        """

        return math.exp(-1 * ((x - mu) ** 2) / (2 * sigma ** 2)) / \
            math.sqrt(2 * math.pi * sigma ** 2)


    # Build basis function set
    d = 4
    simga = 0.5
    min_pos = env.unwrapped.min_position
    max_pos = env.unwrapped.max_position
    step = (max_pos - min_pos) / d
    phi = [
        lambda s: normal(mu, simga, s[0]) for mu in np.arange(
            min_pos + step/2, max_pos + step/2, step
        )
    ]

    llp_irl(S0, k, T, phi)


    # # A sample problem for llp_irl

    # # Sample to build a sub-set of state space
    # n = 100
    # S0 = np.random.normal(0, 5, n)

    # # Define a sampling transition function
    # def T(s, a):
    #     """
    #     A continuous analog of the 2-state discrete MDP above
    #     """
    #     if s < 0:
    #         # In 'state 1'
    #         if a == 0:
    #             # Expert policy from s0
    #             return np.random.choice([-0.5, 0.5], p=[0.4, 0.6])
    #         else:
    #             # Non-expert action from s0
    #             return np.random.choice([-0.5, 0.5], p=[0.9, 0.1])
    #     else:
    #         # In 'state 2'
    #         if a == 0:
    #             # Expert policy from s1
    #             return np.random.choice([-0.5, 0.5], p=[1, 0])
    #         else:
    #             # Non-expert action from s1
    #             return np.random.choice([-0.5, 0.5], p=[1, 0])


    # # Use a set of normal radial basis functions
    # phi = [
    #     lambda s: normal(-0.5, 0.25),
    #     lambda s: normal(0.5, 0.25)
    # ]

    # # Try LLP IRL
    # alpha = rewards, _ = llp_irl(S0, 2, T, phi)

    # # Compose reward function R(s)
    # R = lambda s: np.inner(alpha, [p(s) for p in phi])

    # # Test reward function
    # print([R(-0.5), R(0.5)])


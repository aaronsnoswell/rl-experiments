"""
Defines an interface for a Markov Decision Process policy
"""

import numpy as np
import math


class Policy():
    """
    A policy is a distribution over actions, given the current state
    """

    def __init__(self):
        """
        Constructor
        """

        # A list of parameters that should be set by any sub-class
        self.mdp_type
        self.policy_mapping

        raise NotImplementedError


    def __str__(self):
        """
        Get string representation
        """
        return "<Policy object>"


    def get_action_distribution(self, current_state):
        """
        Returns a distribution {a_i:p_i, ... a_n:p_n} over action
        given the current state
        """
        return self.policy_mapping[current_state]


    def get_action(self, mdp, current_state):
        """
        Returns a sampled action from the given state
        """

        action_distribution = self.get_action_distribution(current_state)

        action_weights = np.array([])
        for action in mdp.get_action_set():
            action_weights = np.append(action_weights, action_distribution.get(action, 0))

        return np.random.choice(mdp.get_action_set(), p=action_weights)


class UniformRandomPolicy(Policy):
    """
    Implements a uniform random distribution over possible actions
    from each state
    """


    def __init__(self, mdp):
        """
        Constructor
        """

        # Store reference to MDP type
        self.mdp_type = type(mdp)

        self.policy_mapping = {}
        
        for state in mdp.get_state_set():

            self.policy_mapping[state] = {}

            # Initialize all actions to 0 preference
            for action in mdp.get_action_set():
                self.policy_mapping[state][action] = 0

            # Apply a uniform distribution to the possible actions
            possible_actions = mdp.get_possible_action_mapping()[state]
            for action in possible_actions:
                self.policy_mapping[state][action] = 1.0 / len(possible_actions)


    def __str__(self):
        """
        Get string representation
        """
        return "<UniformRandomPolicy initialized on {} MarkovDecisionProcess>".format(
            self.mdp_type
        )


def iterative_policy_evaluation(
    mdp,
    policy,
    *,
    initial_value_function=None,
    max_iterations=math.inf,
    on_iteration=None
    ):
    """
    Performs Iterative Policy Evaluation to determine a value function
    under the given policy
    """

    # Initialize the value function
    v = {}
    for state in mdp.state_set:
        v[state] = 0
    
    if initial_value_function is not None:
        v = initial_value_function

    k = 0
    while True:

        # Initialize the temporary value function v_{k+1}
        v_new = {}
        for state in mdp.state_set:
            v_new[state] = v[state]

        for state in mdp.state_set:

            new_value = 0

            state_index = np.where(mdp.state_set == state)[0][0]

            for action in policy.policy_mapping[state]:

                # Look up index of action
                action_index = np.where(mdp.action_set == action)[0][0]

                action_probability = policy.policy_mapping[state][action]
                reward_value = mdp.reward_mapping.get(state, {}).get(action, 0)

                next_state_expectation = 0
                for next_state in mdp.state_set:

                    # Look up index of state
                    next_state_index = np.where(mdp.state_set == next_state)[0][0]
                    
                    # Get probability of transitioning to that state under s, a
                    transition_prob = mdp.transition_matrix[state_index * len(mdp.action_set) + action_index][next_state_index]

                    # Get current estimate of value for that state
                    next_state_expectation += transition_prob * v.get(next_state, 0)

                # Discount the expectation
                next_state_expectation *= mdp.discount_factor

                # Sum with current sate reward
                new_value += action_probability * (reward_value + next_state_expectation)

            # Store new value estimate for this state
            v_new[state] = new_value

        # Update the value function
        v = v_new
        k += 1
        
        if on_iteration is not None:
            on_iteration(k, v)

        # Check termination condition
        if k == max_iterations: break

    return v




            





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
        self.mdp
        self.policy_mapping

        raise NotImplementedError


    def get_action_distribution(self, current_state):
        """
        Returns a distribution {a_i:p_i, ... a_n:p_n} over actions
        given the current state
        """
        return self.policy_mapping[current_state]


    def get_action(
        self,
        current_state,
        *,
        tie_breaker_action=None,
        epsillon=0.001
        ):
        """
        Returns a sampled action from the given state

        If tie_breaker_action is given, this will be used in the event of a
        tie in possible action values, otherwise one action will be sampled
        Epsillon is used for comparing values for equivalentce
        """

        action_distribution = self.get_action_distribution(current_state)
        action_options = np.array(list(action_distribution.keys()))
        action_weights = np.array(list(action_distribution.values()))
        best_action_indices = abs(action_weights - np.max(action_weights)) < epsillon
        best_actions = action_options[best_action_indices]
        best_action_weights = action_weights[best_action_indices]
        num_best_actions = len(best_actions)

        if num_best_actions == 1:

            # Return best action
            return best_actions[0]

        else:

            if tie_breaker_action == None:

                # Sample an action and return it
                return np.random.choice(best_actions, p=best_action_weights)

            else:

                # Return the tie-breaker action
                return tie_breaker_action


    def evaluate(self, initial_value_function):
        """
        Evaluates this policy once to get a new value function estimate
        """

        # Deep copy the value function
        value_function = dict(initial_value_function)

        for state_index, state in enumerate(self.mdp.state_set):

            new_value = 0

            for action_index, action in enumerate(self.policy_mapping[state]):

                action_probability = self.policy_mapping[state][action]
                reward_value = self.mdp.reward_mapping.get(state, {}).get(action, 0)

                next_state_expectation = 0
                for next_state_index, next_state in enumerate(self.mdp.state_set):
                    
                    # Get probability of transitioning to that state under s, a
                    transition_prob = self.mdp.transition_matrix[state_index * len(self.mdp.action_set) + action_index][next_state_index]

                    # Get current estimate of value for that state
                    next_state_expectation += transition_prob * initial_value_function.get(next_state, 0)

                # Discount the expectation
                next_state_expectation *= self.mdp.discount_factor

                # Sum with current sate reward
                new_value += action_probability * (reward_value + next_state_expectation)

            # Store new value estimate for this state
            value_function[state] = new_value

        return value_function


    def iterative_policy_evaluation(
        self,
        initial_value_function,
        *,
        on_iteration=None,
        epsillon=0.001,
        max_iterations=math.inf,
        verbose=False
        ):
        """
        Performs IPE to determine a value function
        """

        # Deep copy the value function
        value_function = dict(initial_value_function)

        k = 0
        continue_iterating = True
        while continue_iterating:
            
            # Update iteration index
            k += 1

            # Update the value function
            new_value_function = self.evaluate(value_function)
            
            if on_iteration is not None:
                on_iteration(k, value_function, new_value_function)

            # Find largest value difference
            largest_delta = 0
            for state in value_function:
                delta = abs(value_function[state] - new_value_function[state])
                if delta > largest_delta:
                    largest_delta = delta

            # Check convergence criteria
            if largest_delta < epsillon:
                if verbose: print("IPE: Got converged value function after {} iteration(s)".format(k))
                continue_iterating = False

            if k >= max_iterations:
                if verbose: print("IPE: Stopping after {} iteration(s)".format(k))
                continue_iterating = False

            value_function = new_value_function

        return value_function


    def __str__(self):
        """
        Get string representation
        """
        return "<{} initialized on {} MarkovDecisionProcess>".format(
            type(self).__name__,
            type(self.mdp)
        )


    def __eq__(self, other):
        # Check these policies are for the same MDP
        if self.mdp != other.mdp: return False

        # Verify that these policies are, in fact, equal
        # Can't use simple ==, because the policie mappings could include
        # floating point state representations
        for from_state in self.policy_mapping:
            
            if from_state not in other.policy_mapping.keys():
                return False

            for to_state in self.policy_mapping[from_state]:

                if to_state not in other.policy_mapping[from_state]:
                    return False

                    if not isclose(self.policy_mapping[from_state][to_state], other.policy_mapping[from_state][to_state]):
                        return False

        # Return true
        return True


    @staticmethod
    def policy_iteration(
        initial_policy,
        initial_value_function,
        *,
        on_iteration=None,
        epsillon=0.001,
        max_value_iterations=math.inf,
        max_iterations=math.inf,
        verbose=False
        ):
        """
        Performs policy iteration to find V*, pi*
        """

        # Deep copy the value function
        policy = initial_policy
        value_function = dict(initial_value_function)

        k = 0
        continue_iterating = True
        while continue_iterating:

            # Increment counter
            k += 1

            # Do policy evaluation until value function converges
            new_value_function = policy.iterative_policy_evaluation(
                value_function,
                epsillon=epsillon,
                max_iterations=max_value_iterations,
                verbose=verbose
            )

            # Do greedy policy improvement
            new_policy = GreedyPolicy(policy.mdp, new_value_function)

            # Call on_iteration callback
            if on_iteration is not None:
                on_iteration(k, value_function, policy, new_value_function, new_policy)

            # Test policy for stability
            if new_policy == policy:
                if verbose: print("PI: Got policy convergence after {} iterations".format(k))
                continue_iterating = False

            if k >= max_iterations:
                if verbose: print("PI: Stopping after {} iterations".format(k))
                continue_iterating = False

            # Copy new functions
            value_function = new_value_function
            policy = new_policy

        # Return converged value function
        return value_function, policy


class UniformPolicy(Policy):
    """
    Implements a policy that always does the same thing, regardless of state
    """


    def __init__(self, mdp, uniform_action):
        """
        Constructor
        """

        # Store reference to MDP
        self.mdp = mdp
        self.policy_mapping = {}
        
        for state in self.mdp.get_state_set():

            self.policy_mapping[state] = {}

            possible_actions = self.mdp.get_possible_action_mapping()[state]
            for action in possible_actions:
                if action == uniform_action:
                    self.policy_mapping[state][action] = 1
                else:
                    self.policy_mapping[state][action] = 0


class UniformRandomPolicy(Policy):
    """
    Implements a uniform random distribution over possible actions
    from each state
    """


    def __init__(self, mdp):
        """
        Constructor
        """

        # Store reference to MDP
        self.mdp = mdp
        self.policy_mapping = {}
        
        for state in mdp.get_state_set():

            self.policy_mapping[state] = {}

            # Apply a uniform distribution to the possible actions
            possible_actions = mdp.get_possible_action_mapping()[state]
            for action in possible_actions:
                self.policy_mapping[state][action] = 1.0 / len(possible_actions)


class GreedyPolicy(Policy):
    """
    Implements a greedy policy - goes for the highest estimated value at each
    state
    """


    def __init__(self, mdp, value_function, epsillon=0.001):
        """
        Constructor
        """

        # Store reference to MDP
        self.mdp = mdp
        self.policy_mapping = {}
        
        for state in mdp.get_state_set():

            self.policy_mapping[state] = {}

            # Initialize all actions to 0 preference
            for action in mdp.get_action_set():
                self.policy_mapping[state][action] = 0

            # Find the set of best possible actions
            possible_actions = mdp.get_possible_action_mapping()[state]

            if len(possible_actions) == 0:
                # If we can't do anything from this state, skip
                continue

            possible_action_values = np.array([
                    value_function[
                        mdp.transition(state, action)[2]
                    ] for action in possible_actions
                ])
            best_action_indices = abs(
                    possible_action_values - possible_action_values[
                        np.argmax(possible_action_values)
                    ]
                ) < epsillon
            best_actions = np.array(possible_actions)[best_action_indices]

            # Assign transition preferences
            for best_action in best_actions:
                self.policy_mapping[state][best_action] = 1 / len(best_actions)


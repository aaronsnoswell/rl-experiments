"""
Defines an interface for a Markov Decision Process
"""

import numpy as np

from .markov_process import MarkovProcess
from .markov_reward_process import MarkovRewardProcess


class MarkovDecisionProcess(MarkovRewardProcess):
    """
    A Markov Decision Process is a tuple <S, A, P, R, gamma>
    S: Finite set of states
    A: Finite set of possible actions
    P: State-action transition matrix
    R: State-action reward function
    gamma: Discount factor
    """

    def __init__(self):
        """
        Constructor
        """
        raise NotImplementedError


    def __str__(self):
        return "<MarkovDecisionProcess(\n  S: {}\n  A: {}\n  P: {}\n  R: {}\n  gamma: {}\n)>".format(
            str(self.state_set).replace("\n", "\n     "),
            str(self.action_set).replace("\n", "\n     "),
            str(self.transition_matrix).replace("\n", "\n     "),
            str(self.reward_mapping).replace("\n", "\n     "),
            self.discount_factor
        )


    def get_reward_mapping(self):
        """
        Returns the reward mapping {s_i: r_i}
        """
        return self.reward_mapping


    def get_reward_vector(self):
        """
        Returns the reward vector [r_i, ...] for every state s_i
        in the ordered list state_set
        """
        raise NotImplementedError


    def solve_bellman_equation(self, policy):
        """
        Solves the Bellman equation for the MRP giving
        v = ((I - discount*P)^-1) * R
        This is only feasible for small processes
        """
        raise NotImplementedError


    def get_expected_reward(self, current_state):
        """
        Returns the expected reward at time t+1 given we are currently in the given state s_t
        """
        raise NotImplementedError


    def get_value(self, current_state, policy, *, num_rollouts=1000, max_length=None):
        """
        Computes an expectation of return up to horizon max_length from the current state
        """
        raise NotImplementedError


    def get_return(self, current_state, policy, *, max_length=None):
        """
        Rolls out the MRP once from the given state and calculates the return
        """
        raise NotImplementedError



    def get_value_map(self, policy, *, num_rollouts=1000, max_length=None):
        """
        Performs many rollouts to compute an estimate of the value function
        """
        raise NotImplementedError


    def rollout(self, current_state, policy, *, max_length=None):
        """
        Returns a single rollout of the process [S, S', S'', ..., S_terminal]
        """
        raise NotImplementedError



    def transition(self, current_state, action):
        """
        Returns the reward for being in the current state, and a subsequent state
        """
        raise NotImplementedError


    @staticmethod
    def from_dict(markov_decision_process_dict):
        """
        Converts a dictionary {s: a: {s': p_s', ...}, ...}, ...} to
        a set of states [s, s', ...], a set of actions [a, ...], 
        a state transition matrix, and a set of terminal states [s_t, ...]
        """

        # Build state and action sets
        state_set = list(markov_decision_process_dict.keys())
        action_set = []

        for state in markov_decision_process_dict:

            for action in markov_decision_process_dict[state]:

                if action not in action_set:
                    action_set.append(action)

                for subsequent_state in markov_decision_process_dict[state][action]:

                    if subsequent_state not in state_set:
                        state_set.append(subsequent_state)

        state_set = np.array(state_set)
        action_set = np.array(action_set)

        # Initialize terminal state set
        terminal_state_set = np.array([])

        # Build transition matrix
        transition_matrix = np.zeros(
            shape=(len(state_set) * len(action_set), len(state_set))
        )

        # For every 'from-state'
        for i in range(len(state_set)):
            state = state_set[i]

            if state not in markov_decision_process_dict:
                # This state is terminal

                for a in range(len(action_set)):
                    # Set all state-actions from this state
                    # to return to this state
                    transition_matrix[i*len(action_set)+a, i] = 1
                terminal_state_set = np.append(state, terminal_state_set)
                continue

            # Loop over all actions
            for a in range(len(action_set)):
                action = action_set[a]

                if action not in markov_decision_process_dict[state]:
                    # This action is not available from this state
                    transition_matrix[i*len(action_set) + a, i] = 1
                    continue

                # Loop over subsequent states
                for j in range(len(state_set)):

                    subsequent_state = state_set[j]

                    if subsequent_state not in markov_decision_process_dict[state][action]:
                        # This state not reachable from the current state and chosen action
                        continue

                    probability = markov_decision_process_dict[state][action][subsequent_state]
                    print(state, action, subsequent_state, probability, "({},{})".format(i*len(action_set) + a, j))
                    transition_matrix[i*len(action_set) + a, j] = probability


        return state_set, action_set, transition_matrix, terminal_state_set

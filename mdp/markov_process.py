"""
Defines an interface for a Markov Process
"""

import numpy as np


class MarkovProcess():
    """
    A Markov Process is a tuple <S, P>
    S: Finite set of states
    P: State transition matrix
    """

    def __init__(self):
        """
        Constructor
        """
        raise NotImplementedError


    def get_state_set():
        """
        Returns the set of states
        """
        raise NotImplementedError


    def get_state_transition_matrix():
        """
        Returns the state transition matrix
        """
        raise NotImplementedError


    @staticmethod
    def from_dict(markov_process_dict):
        """
        Converts a dictionary {s: {s: p, s': p_s', ...}, ...} to
        a set of states [s, s', ...] and a state transition matrix
        """

        # Build state set
        state_set = list(markov_process_dict.keys())
        for state in markov_process_dict:
            for subsequent_state in markov_process_dict[state]:
                if subsequent_state not in state_set:
                    state_set.append(subsequent_state)
        state_set = np.array(state_set)

        # Build transition matrix
        transition_matrix = np.zeros(
            shape=(len(state_set), len(state_set))
        )

        # For every 'from-state'
        for i in range(len(state_set)):
            state = state_set[i]

            if state not in markov_process_dict:
                # This state is terminal
                transition_matrix[i, i] = 1
                continue

            # Get transition probabilities for this state
            probs = markov_process_dict[state]

            # For every 'to-state'
            for j in range(len(state_set)):
                subsequent_state = state_set[j]
                if subsequent_state not in probs:
                    # This subsequent state is unreachable from the current state
                    continue

                # Store the transition probability
                transition_matrix[i, j] = probs[subsequent_state]




        return state_set, transition_matrix

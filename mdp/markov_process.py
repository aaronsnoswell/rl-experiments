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


    def rollout(self, current_state, *, max_length=None):
        """
        Returns a single rollout of the process [S, S', S'', ..., S_termainl]
        """
        assert current_state in self.state_set, \
            "Given state is not in state set"

        curr = current_state
        history = np.array([current_state])
        while curr not in self.terminal_state_set:
            
            if max_length is not None:
                if len(history) >= max_length: break

            curr = self.transition(curr)
            history = np.append(history, curr)

        return history


    def transition(self, current_state):
        """
        Returns a subsequent state
        """
        assert current_state in self.state_set, \
            "Given state is not in state set"

        index = np.where(self.state_set == current_state)[0][0]
        return np.random.choice(
            self.state_set,
            p=self.state_transition_matrix[index, :]
        )


    def get_state_set(self):
        """
        Returns the set of all states
        """
        raise NotImplementedError


    def get_terminal_state_set(self):
        """
        Returns the set of terminal states
        """
        raise NotImplementedError


    def get_state_transition_matrix(self):
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

        # Initialize terminal state set
        terminal_state_set = np.array([])

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
                terminal_state_set = np.append(state, terminal_state_set)
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


        return state_set, transition_matrix, terminal_state_set

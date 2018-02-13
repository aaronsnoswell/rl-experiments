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

        # A list of parameters that should be set by any sub-class
        self.state_set
        self.terminal_state_set
        self.transition_matrix

        raise NotImplementedError


    def __str__(self):
        """
        Get string representation
        """
        return "<MarkovProcess(\n  S: {}\n  P: {}\n)>".format(
            str(self.state_set).replace("\n", "\n     "),
            str(self.transition_matrix).replace("\n", "\n     ")
        )


    def rollout(self, current_state, *, max_length=None):
        """
        Returns a single rollout of the process [S, S', S'', ..., S_terminal]
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
            p=self.transition_matrix[index, :]
        )


    def get_state_set(self):
        """
        Returns the set of all states
        """
        return self.state_set


    def get_terminal_state_set(self):
        """
        Returns the set of terminal states
        """
        return self.terminal_state_set


    def get_transition_matrix(self):
        """
        Returns the state transition matrix
        """
        return self.transition_matrix


    def compute_stationary_distribution(self, *, num_rollouts=10000, max_length=None):
        """
        Estimates the stationary distribution of a process
        """

        print("Estimating the stationary distribution with {} rollouts".format(num_rollouts))
        print("(this may take a while)")
        
        state_counts = {}
        for state in self.state_set:
            state_counts[state] = 0

        total_visited_states = 0
        for n in range(num_rollouts):
            # Pick a starting state
            start_state = np.random.choice(self.state_set)

            # Do a full rollout
            rollout = self.rollout(start_state, max_length=max_length)
            total_visited_states += len(rollout)

            # Add up the states we visited
            for visited_sate in rollout:
                state_counts[visited_sate] += 1

        # Convert to a probability
        stationary_distribution = []
        for state in state_counts:
            stationary_distribution.append(state_counts[state] / total_visited_states)

        return np.array(stationary_distribution)



    @staticmethod
    def from_dict(markov_process_dict):
        """
        Converts a dictionary {s: {s': p_s', ...}, ...} to
        a set of states [s, s', ...], a state transition matrix,
        and a set of terminal states [s_t, ...]
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

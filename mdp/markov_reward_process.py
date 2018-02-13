"""
Defines an interface for a Markov Reward Process
"""

import numpy as np

from .markov_process import MarkovProcess


class MarkovRewardProcess(MarkovProcess):
    """
    A Markov Reward Process is a tuple <S, P, R, gamma>
    S: Finite set of states
    P: State transition matrix
    R: State reward function
    gamma: Discount factor
    """

    def __init__(self):
        """
        Constructor
        """
        raise NotImplementedError


    def __str__(self):
        """
        Get string representation
        """
        return "<MarkovRewardProcess(\n  S: {}\n  P: {}\n  R: {}\n  gamma: {}\n)>".format(
            str(self.state_set).replace("\n", "\n     "),
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
        reward_vector = np.array([])
        for state in self.state_set:
            reward_vector = np.append(reward_vector, self.reward_mapping[state])
        return reward_vector


    def solve_bellman_equation(self):
        """
        Solves the Bellman equation for the MRP giving
        v = ((I - discount*P)^-1) * R
        This is only feasible for small processes
        """

        assert self.discount_factor != 1, \
            "Cannot solve bellman equation for infinitely far-sighted process (results in a singular matrix inversion)"

        value_vector = np.matmul(
            np.linalg.inv(
                np.identity(
                    len(self.state_set)
                ) - self.discount_factor * self.transition_matrix
            ),
            self.get_reward_vector()
        )

        self.value_map = {}
        for i in range(len(self.state_set)):
            self.value_map[self.state_set[i]] = value_vector[i]

        return self.value_map



    def get_discount_factor(self):
        """
        Returns the discount factor
        """
        return self.discount_factor


    def get_expected_reward(self, current_state):
        """
        Returns the expected reward at time t+1 given we are currently in the given state s_t
        """

        assert current_state in self.state_set, \
            "Given state is not in state set"

        assert current_state in self.reward_mapping, \
            "Given state is not in reward mapping"

        return self.reward_mapping[current_state]


    def get_value(self, current_state, *, num_rollouts=1000, max_length=None):
        """
        Computes an expectation of return up to horizon max_length from the current state
        """

        assert current_state in self.state_set, \
            "Given state is not in state set"

        value = 0
        for i in range(num_rollouts):
            value += self.get_return(current_state, max_length=max_length)
        value /= num_rollouts

        return value


    def get_return(self, current_state, *, max_length=None):
        """
        Rolls out the MRP once from the given state and calculates the return
        """

        assert current_state in self.state_set, \
            "Given state is not in state set"

        # Perform rollout
        history = self.rollout(current_state, max_length=max_length)
        
        # Slice record array to get rewards
        rewards = history['reward']

        # Remove None types (e.g. initial state has reward of None)
        rewards = rewards[rewards != None]

        # Apply discount factor
        discounted_rewards = np.empty_like(rewards)
        for i in range(len(rewards)):
            reward = rewards[i]
            discounted_rewards[i] = reward * self.discount_factor ** i

        return np.sum(discounted_rewards)


    def get_value_map(self, *, num_rollouts=1000, max_length=None):
        """
        Performs many rollouts to compute an estimate of the value function
        """

        print(
            "Computing value function with {} rollouts, discount {} and max length {}".format(
                num_rollouts,
                self.discount_factor,
                max_length
            )
        )
        print("(this may take a while...)")

        self.value_map = {}
        for state in self.state_set:
            self.value_map[state] = self.get_value(
                state,
                num_rollouts=num_rollouts,
                max_length=max_length
            )

        return self.value_map


    def rollout(self, current_state, *, max_length=None):
        """
        Returns a single rollout of the process [(R, S), (R', S'), ..., (R_terminal, S_terminal)]
        """

        assert current_state in self.state_set, \
            "Given state is not in state set"

        curr = (None, current_state)

        history = np.array(
            [curr],
            dtype=[
                ('reward', np.array(self.reward_mapping.values()).dtype),
                ('state', np.array(self.state_set).dtype)
            ]
        )

        while curr[1] not in self.terminal_state_set:

            if max_length is not None:
                if len(history) >= max_length: break

            curr = self.transition(curr[1])
            history = np.append(
                history,
                np.array([curr], dtype=history.dtype)
            )

        return history


    def transition(self, current_state):
        """
        Returns the reward for being in the current state, and a subsequent state
        """

        assert current_state in self.state_set, \
            "Given state is not in state set"

        reward = self.get_expected_reward(current_state)
        new_state = super().transition(current_state)
        return (reward, new_state)


    def decompose(self, policy):
        """
        Decomposes this MRP to a Markov Process by discarding the reward terms
        See https://stackoverflow.com/a/18403354/885287 for technical method to
        dynamically define a sub-class

        TODO ajs 18/Feb/2018 implement this
        """
        raise NotImplementedError

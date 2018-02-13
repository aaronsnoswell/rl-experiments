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

        # A list of parameters that should be set by any sub-class
        self.state_set
        self.terminal_state_set
        self.action_set
        self.transition_matrix
        self.reward_mapping
        self.possible_action_mapping
        self.discount_factor

        raise NotImplementedError


    def __str__(self):
        """
        Get string representation
        """
        return "<MarkovDecisionProcess(\n  S: {}\n  A: {}\n  P: {}\n  R: {}\n  gamma: {}\n)>".format(
            str(self.state_set).replace("\n", "\n     "),
            str(self.action_set).replace("\n", "\n     "),
            str(self.transition_matrix).replace("\n", "\n     "),
            str(self.reward_mapping).replace("\n", "\n     "),
            self.discount_factor
        )


    def get_action_set(self):
        """
        Returns the set of all possible actions {a_i, ...}
        """
        return self.action_set


    def get_reward_mapping(self):
        """
        Returns the reward mapping {s_i: {a_i: r_i, ...}, ...}
        """
        return self.reward_mapping


    def get_possible_action_mapping(self):
        """
        Returns a mapping indicating which actions are available
        from each state
        """
        return self.possible_action_mapping


    def get_reward_vector(self):
        """
        Returns the reward vector [r_i, ...] for every state-action pair (s_i, a_i)
        """
        reward_vector = np.array([])
        for state in self.state_set:
            for action in self.action_set:
                reward_vector = np.append(reward_vector, self.get_expected_reward(state, action))

        return reward_vector


    def solve_bellman_equation(self, policy):
        """
        Solves the Bellman equation for the MRP giving
        v = ((I - discount*P)^-1) * R
        This is only feasible for small processes
        """
        raise NotImplementedError


    def get_expected_reward(self, current_state, action):
        """
        Returns the expected reward at time t+1 given we are currently in the given state s_t
        and take action a_t
        """

        assert current_state in self.state_set, \
            "Given state ({}) is not in state set".format(current_state)

        assert current_state in self.reward_mapping, \
            "Given state ({}) is not in reward mapping".format(current_state)

        return self.reward_mapping[current_state].get(action, None)


    def get_value(self, current_state, policy, *, num_rollouts=10000, max_length=None):
        """
        Computes an expectation of return up to horizon max_length from the current state
        """

        assert current_state in self.state_set, \
            "Given state ({}) is not in state set".format(current_state)

        value = 0
        for i in range(num_rollouts):
            value += self.get_return(current_state, policy, max_length=max_length)
        value /= num_rollouts

        return value


    def get_return(self, current_state, policy, *, max_length=None):
        """
        Rolls out the MRP once from the given state and calculates the return
        """

        assert current_state in self.state_set, \
            "Given state ({}) is not in state set".format(current_state)

        # Perform rollout
        history = self.rollout(current_state, policy, max_length=max_length)
        
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


    def get_value_map(self, policy, *, num_rollouts=10000, max_length=None):
        """
        Performs many rollouts to compute an estimate of the state-value function
        """

        print(
            "Computing state-value function with {} rollouts, discount {} and max length {}".format(
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
                policy,
                num_rollouts=num_rollouts,
                max_length=max_length
            )

        return self.value_map


    def rollout(self, current_state, policy, *, max_length=None):
        """
        Returns a single rollout of the process [(A, R, S), (A', R', S'), ..., (A_terminal, R_terminal, S_terminal)]
        """

        assert current_state in self.state_set, \
            "Given state ({}) is not in state set".format(current_state)

        curr = (None, None, current_state)

        history = np.array(
            [curr],
            dtype=[
                ('action', np.array(self.action_set).dtype),
                ('reward', np.array(list(self.reward_mapping.values())[0].values()).dtype),
                ('state', np.array(self.state_set).dtype)
            ]
        )

        while curr[2] not in self.terminal_state_set:

            state = curr[2]

            if max_length is not None:
                if len(history) >= max_length: break

            # Choose an action from the policy
            action = policy.get_action(self, state)

            # Transition
            curr = self.transition(state, action)

            # Update history
            history = np.append(
                history,
                np.array([curr], dtype=history.dtype)
            )

        return history


    def transition(self, current_state, action):
        """
        Returns the current action, a reward for taking the given action from the
        current state, and a subsequent state
        NB: The subsequent state could be different to where the action leads if
        the environment intervenes
        """

        assert current_state in self.state_set, \
            "Given state ({}) is not in state set".format(current_state)

        reward = self.get_expected_reward(current_state, action)
        state_index = np.where(self.state_set == current_state)[0][0]
        action_index = np.where(self.action_set == action)[0][0]

        new_state = np.random.choice(
            self.state_set,
            p=self.transition_matrix[state_index * len(self.action_set) + action_index, :]
        )

        return (action, reward, new_state)


    def compute_stationary_distribution(self, policy, *, num_rollouts=10000, max_length=None):
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
            rollout = self.rollout(start_state, policy, max_length=max_length)
            total_visited_states += len(rollout)

            # Add up the states we visited
            for action_taken, got_reward, visited_sate in rollout:
                state_counts[visited_sate] += 1

        # Convert to a probability
        stationary_distribution = []
        for state in state_counts:
            stationary_distribution.append(state_counts[state] / total_visited_states)

        return np.array(stationary_distribution)


    def decompose(self, policy):
        """
        Decomposes this MDP, in addition with the given policy, to a MRP
        """

        def mrp_init(self, parent_mdp, policy):
            """
            Decomposes a given MDP and policy into a latent MRP that
            approximates the MDP/Policy combination
            """
            print("Initializing derived MRP from MDP")

            # Set MDP parameters
            self.state_set = parent_mdp.state_set
            self.terminal_state_set = parent_mdp.terminal_state_set
            self.discount_factor = parent_mdp.discount_factor

            # Decompose the transition matrix based on the policy
            self.transition_matrix = np.identity(len(self.state_set))

            # Loop over all states
            for si, state in enumerate(self.state_set):

                if state in self.terminal_state_set: continue

                # Prepare one row in the new transition matrix
                transition_matrix_row = np.zeros(shape=len(self.state_set))

                # Query the policy for a distribution over actions
                action_distribution = policy.get_action_distribution(state)
                for action in action_distribution:

                    # Get an action index
                    ai = np.where(parent_mdp.action_set == action)[0][0]

                    # Get the preference/probability of the policy choosing this action
                    pref_for_action = action_distribution[action]

                    # Accumulate the true transition probabilities
                    transition_matrix_row += pref_for_action * parent_mdp.transition_matrix[si * len(parent_mdp.action_set) + ai, :]

                self.transition_matrix[si, :] = transition_matrix_row

            # Decompose the reward mapping based on the policy
            self.reward_mapping = {}
            for si, state in enumerate(self.state_set):

                # Prepare one entry in the reward mapping
                self.reward_mapping[state] = 0

                if state not in list(parent_mdp.reward_mapping.keys()):
                    continue

                # Query the policy for a distribution over actions
                action_distribution = policy.get_action_distribution(state)
                for action in action_distribution:

                    # Get an action index
                    ai = np.where(parent_mdp.action_set == action)[0][0]

                    # Get the preference/probability of the policy choosing this action
                    pref_for_action = action_distribution[action]

                    # Acuumulate the true expected reward for being in this state
                    expected_reward = parent_mdp.get_expected_reward(state, action)
                    if expected_reward is not None:
                        self.reward_mapping[state] += expected_reward


        dynamic_class_type = type("DerivedMarkovRewardProcess", (MarkovRewardProcess, ), {'__init__': mrp_init})
        return dynamic_class_type(self, policy)


    @staticmethod
    def from_dict(markov_decision_process_dict):
        """
        Converts a dictionary {s: {a: {s': p_s', ...}, ...}, ...} to
        a set of states [s, s', ...], a set of actions [a, ...], 
        a state-action to state transition matrix, a mapping from states to possible actions
        {s: [a1, a2], ...} and a set of terminal states [s_t, ...]
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
                    transition_matrix[i*len(action_set) + a, j] = probability

        # Build set of possible action mappings
        possible_action_mapping = {}
        for state in state_set:

            possible_action_mapping[state] = []
            if state not in markov_decision_process_dict: continue

            for action in markov_decision_process_dict[state]:
                possible_action_mapping[state].append(action)


        return state_set, action_set, transition_matrix, possible_action_mapping, terminal_state_set

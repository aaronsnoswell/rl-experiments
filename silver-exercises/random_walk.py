"""
A simple python script to demonstrate a Markov Decision Process problem
"""

import numpy as np


# Markov state, probability and reward definitions for the random walk problem
rw_states = ['-3', '-2', '-1', '0', '1', '2', '3']

rw_policy = {
    '-2': np.array([['l', 'r'],
                   [ 0.5, 0.5]]),
    '-1': np.array([['l', 'r'],
                   [ 0.5, 0.5]]),
    '0': np.array([['l', 'r'],
                   [ 0.5, 0.5]]),
    '1': np.array([['l', 'r'],
                   [ 0.5, 0.5]]),
    '2': np.array([['l', 'r'],
                   [ 0.5, 0.5]])
}

# NB: State-action pairs with no transition probabilities are assumed terminal
rw_state_action_probs = {
    '-2': {
        'l' : np.array([['-3'], [1]]),
        'r' : np.array([['-1'], [1]])
    },
    '-1': {
        'l' : np.array([['-2'], [1]]),
        'r' : np.array([['0'], [1]])
    },
    '0': {
        'l' : np.array([['-1'], [1]]),
        'r' : np.array([['1'], [1]])
    },
    '1': {
        'l' : np.array([['0'], [1]]),
        'r' : np.array([['2'], [1]])
    },
    '2': {
        'l' : np.array([['1'], [1]]),
        'r' : np.array([['3'], [1]])
    }
}

# NB: States with no transition probabilities are assumed terminal
rw_state_probs = {
    '-2': np.array([['-3', '-1'],
                   [ 0.5,  0.5]]),
    '-1': np.array([['-2', '0'],
                   [ 0.5,  0.5]]),
    '0': np.array([['-1', '1'],
                   [ 0.5,  0.5]]),
    '1': np.array([['0', '2'],
                   [ 0.5,  0.5]]),
    '2': np.array([['1', '3'],
                   [ 0.5,  0.5]])
}

# NB: States with no defined reward are assumed to have 0 reward
rw_rewards = {
    '-3': -1,
    '3': 1
}


"""
A simple Markov Process class
"""
class MarkovProcess(object):


    """
    Constructor
    """
    def __init__(self, states, state_transition_probabilities, initial_state=None):

        # Store states
        self.states = states

        # Store transition probabilities
        self.state_transition_probabilities = state_transition_probabilities

        # Store initial and current state
        self.initial_state = self.states[0]
        if initial_state is not None:
            self.initial_state = initial_state
        self.state = self.initial_state

        # Initialize number of steps taken
        self.num_steps = 0

        # Compute list of terminal states
        self.terminal_states = []
        for state in self.states:
            if state in self.state_transition_probabilities.keys():
                p = self.state_transition_probabilities[state]
                if len(p[0]) == 1 and p[0][0] == state:
                    # If a state transitions only to itself, it is a terminal state
                    self.terminal_states.append(state)
            else:
                # If a state has no probabilities defined, treat it as terminal
                self.terminal_states.append(state)
    

    def __repr__(self):
        states = '(' + ', '.join(self.states) + ')'
        probs = "(...)"

        return "Markov Process with S={}, Ps={}, S_t={:+d}".format(
            states,
            probs,
            int(self.state)
        )


    """
    Steps us to a new state
    """
    def step(self):

        # Transition to new state
        tmp = self.state_transition_probabilities[self.state]
        self.state = np.random.choice(a=list(tmp[0]), p=list(tmp[1]))

        # Increment step counter
        self.num_steps += 1


    """
    Lazy check to see if the process has reached a terminal state yet
    """
    def is_terminated(self):
        return self.state in self.terminal_states


    """
    Performs a full rollout of the given Markov Process
    """
    def rollout(self, verbose=False):
        # Perform a full rollout of the Markov Process
        while not self.is_terminated():
            self.step()
            if verbose is True:
                print(self)


    """
    Resets the Markov Process to it's initial state and return
    """
    def reset(self):
        self.state = self.initial_state
        self.num_steps = 0


"""
A simple Markov Reward Process class

TODO ajs 24/Jan/2017 Add ability to do future return estimation using discount
"""
class MarkovRewardProcess(MarkovProcess):


    """
    Constructor
    """
    def __init__(self, states, state_transition_probabilities, rewards=None, discount=1, initial_state=None):

        # Call Markov Process constructor
        super().__init__(states, state_transition_probabilities, initial_state)

        # Store states
        self.rewards = rewards

        # Store discount
        self.discount = discount

        # Initialize return
        self._return = 0


    def __repr__(self):
        states = '(' + ', '.join(self.states) + ')'
        probs = "(...)"
        rewards = "(...)"

        return "Markov Reward Process with S={}, Psa={}, R={}, g={:.2f}, S_t={:+d}, G_t={:+d}".format(
            states,
            probs,
            rewards,
            self.discount,
            int(self.state),
            self._return
        )


    """
    Steps us to a new state
    """
    def step(self):
        
        # Call Markov Process step
        super().step()

        # Check if we got any reward
        if self.rewards is not None:
            self._return += self.rewards.get(self.state, 0)


    """
    Resets the Markov Process to it's initial state and return
    """
    def reset(self):
        
        # Call Markov Process reset
        super().reset()

        self._return = 0


"""
A simple Markov Decision Process class
"""
class MarkovDecisionProcess(MarkovRewardProcess):


    """
    Constructor
    """
    def __init__(
        self,
        states,
        policy,
        state_action_transition_probabilities,
        rewards=None,
        discount=1,
        initial_state=None
        ):

        # TODO ajs 24/Jan/2017 Unroll the decision process under the given policy into a
        # reward process, then call Markov Reward Process constructor
        #super().__init__(states, transition_probabilities, initial_state)
        pass


    def __repr__(self):
        states = '(' + ', '.join(self.states) + ')'
        policy = "(...)"
        probs = "(...)"
        rewards = "(...)"

        return "Markov Decision Process with S={}, pi={}, Psa={}, R={}, g={:.2f}, S_t={:+d}, G_t={:+d}".format(
            states,
            policy,
            probs,
            rewards,
            self.discount,
            int(self.state),
            self._return
        )


    """
    Steps us to a new state
    """
    def step(self):
        # TODO: Sample policy, then let environment go
        pass




"""
Main function
"""
def main():

    #mp = MarkovProcess(rw_states, rw_state_probs, initial_state='0')
    mp = MarkovRewardProcess(rw_states, rw_state_probs, rewards=rw_rewards, initial_state='0')

    average_num_steps = 0
    average_return = 0
    N = 10
    for i in range(N):
        mp.reset()
        mp.rollout(True)
        average_num_steps += mp.num_steps
        average_return += mp._return

    average_num_steps /= N
    average_return /= N

    print(average_num_steps)
    print(average_return)


if __name__ == "__main__":
    main()


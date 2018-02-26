
import numpy as np
from contextlib import contextmanager

from .markov_decision_process import MarkovDecisionProcess


@contextmanager
def show_complete_array():
    oldoptions = np.get_printoptions()
    np.set_printoptions(threshold=np.inf)
    yield
    np.set_printoptions(**oldoptions)



class GridWorld(MarkovDecisionProcess):
    """
    Implements a grid world MDP
    """

    # Simple action sets that GridWorld will recognise
    ActionSetCompassFour = ['N', 'E', 'S', 'W']
    ActionSetCompassEight = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']

    class GridWorldState():
        """
        Container class for a grid world state
        """

        def __init__(self, x, y, data):
            """
            Constructor
            """
            self.x = x
            self.y = y
            self.data = data


        def peek(self, action, boundary_result="nothing"):
            """
            Returns the x, y location we would reach given the given action
            """
            x, y = self.x, self.y
            if action == "N":
                y -= 1
            elif action == "NE":
                x += 1
                y -= 1
            elif action == "E":
                x += 1
            elif action == "SE":
                x += 1
                y += 1
            elif action == "S":
                y += 1
            elif action == "SW":
                x -= 1
                y += 1
            elif action == "W":
                x -= 1
            elif action == "NW":
                x -= 1
                y -= 1
            else:
                assert False,\
                    "Error: Unrecognised action {}".format(action)

            return x, y


        def __str__(self):
            """
            Get string representation
            """
            return "<GridWorldState({}, {}, {})>".format(
                str(self.x),
                str(self.y),
                str(self.data)
            )


        def __repr__(self):
            """
            Get machine representation
            """
            return self.__str__()


    def __init__(
        self,
        state_set,
        terminal_state_set,
        action_set,
        transition_matrix,
        reward_mapping,
        possible_action_mapping,
        discount_factor
        ):
        """
        Constructor
        """

        self.state_set = state_set
        self.terminal_state_set = terminal_state_set
        self.action_set = action_set
        self.transition_matrix = transition_matrix
        self.reward_mapping = reward_mapping
        self.possible_action_mapping = possible_action_mapping
        self.discount_factor = discount_factor


    @staticmethod
    def dict_as_grid(the_dict):
        """
        Helper method to convert a dict of GridWorldState to a grid
        """

        # Determine bounds
        width = 0
        height = 0
        for gws in the_dict:
            if gws.x+1 > width: width = gws.x+1
            if gws.y+1 > height: height = gws.y+1

        # Initialize grid
        grid = [[None for x in range(width)] for y in range(height)]

        # Store elements in grid
        for gws in the_dict:
            grid[gws.y][gws.x] = the_dict[gws]

        return np.array(grid)



    @staticmethod
    def from_array(
        gridworld_array,
        state_is_terminal,
        *,
        action_set=ActionSetCompassFour,
        boundary_result="nothing",
        discount_factor=1,
        timestep_reward=-1,
        terminal_reward=10,
        wind_prob=0
    ):
        """
        Converts a grid of states to a gridworld MDP
        @param gridworld_array A 2D array of states
        @param terminal_state A boolean function that indicates if a given
               state is terminal

        @param action_set      (Optional) A set of allowable actions the agent
               can take
        @param boundary_result (Optional) Result of trying to move past the
               boundary. One of "nothing" or "wrap" or "disallow"
        @param discount_factor (Optional) Discount factor
        @param timestep_reward (Optional) Reward per timestep
        @param terminal_reward (Optional) Reward upon terminating
        @param wind_prob       (Optional) Probability that the 'wind' will
               take you to a random state regardless of your action
        """

        # Prepare state sets
        state_set = []
        terminal_state_set = []

        # Prepare transition matrix
        tmp = np.array(gridworld_array)
        height = tmp.shape[0]
        width = tmp.shape[1]
        num_states = width * height
        transition_matrix = np.zeros(
            shape=(num_states * len(action_set), num_states)
        )

        # Prepare reward mapping
        reward_mapping = {}

        # Prepare possible action mapping
        possible_action_mapping = {}


        def apply_boundary(x, y):
            """
            Helper function to apply a boundary result
            """
            if boundary_result == "nothing":
                x = min(max(0, x), width-1)
                y = min(max(0, y), height-1)
            elif boundary_result == "wrap":
                x = x % width
                y = y % height
            elif boundary_result == "disallow":
                assert x >= 0 and x < width and y >= 0 and y < height,\
                    "Error: boundary_result is disallow, but tried to move past boundary"
            else:
                assert False,\
                    "Error: Unrecognised boundary_result {}".format(boundary_result)
            return x, y


        def xy_to_index(x, y):
            """
            Helper function to convert an xy location to an index
            """
            return y * width + x


        # Populate things
        state_index = 0
        for y, row in enumerate(gridworld_array):
            for x, s in enumerate(row):

                state = GridWorld.GridWorldState(x, y, s)

                # Initialize possible action mapping
                possible_action_mapping[state] = []

                # Add to state sets
                state_set.append(state)
                if state_is_terminal(state.data):
                    terminal_state_set.append(state)
                else:
                    # If a state isn't terminal, add all actions as possibliities
                    possible_action_mapping[state] = action_set[:]

                    # Check for boundary disallow result
                    if boundary_result == "disallow":
                        if x == 0:
                            possible_action_mapping[state].remove('W')
                        elif x == width-1:
                            possible_action_mapping[state].remove('E')
                        
                        if y == 0:
                            possible_action_mapping[state].remove('N')
                        elif y == height-1:
                            possible_action_mapping[state].remove('S')

                # Create reward mapping
                reward_mapping[state] = {}

                # Set the default transition matrix entries for this state
                for action_index, action in enumerate(action_set):
                    transition_matrix[state_index * len(action_set) + action_index] = [0] * num_states
                    transition_matrix[state_index * len(action_set) + action_index][state_index] = 1

                for action in possible_action_mapping[state]:

                    # Get the index of this action
                    action_index = action_set.index(action)

                    # Clear this transition matrix entry
                    transition_matrix[state_index * len(action_set) + action_index] = [0] * num_states

                    # Set probability for our target state
                    tx, ty = state.peek(action)
                    tx, ty = apply_boundary(tx, ty)
                    transition_matrix[state_index * len(action_set) + action_index][xy_to_index(tx, ty)] = 1 - wind_prob

                    # Set the probability for all other states the wind could take us to
                    for wind_action_index, wind_action in enumerate(possible_action_mapping[state]):
                        wx, wy = state.peek(wind_action)
                        wx, wy = apply_boundary(wx, wy)
                        transition_matrix[state_index * len(action_set) + action_index][xy_to_index(wx, wy)] += wind_prob / len(action_set)

                    # Update reward mapping
                    reward_mapping[state][action] = 0
                    reward_mapping[state][action] += timestep_reward
                    if(state_is_terminal(gridworld_array[ty][tx])):
                        reward_mapping[state][action] += terminal_reward

                state_index += 1


        return GridWorld(
            np.array(state_set),
            np.array(terminal_state_set),
            np.array(action_set),
            transition_matrix,
            reward_mapping,
            possible_action_mapping,
            discount_factor
        )


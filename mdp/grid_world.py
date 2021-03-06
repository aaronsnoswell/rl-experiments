
import numpy as np
import matplotlib.pyplot as plt

from .markov_decision_process import MarkovDecisionProcess


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

        # Determine grid world bounds
        self.width = 0
        self.height = 0
        for state in self.state_set:
            if state.x+1 > self.width: self.width = state.x+1
            if state.y+1 > self.height: self.height = state.y+1


    def generate_figure(
        self,
        *,
        title=None,
        subtitle=None,
        value_function=None,
        policy=None,
        show_value_text=True
        ):
        """
        Renders a GridWorld, with an optional policy and value function
        
        Creates a new figure window if none exists, otherwise draws to the
        current figure and axes
        """

        # Render settings
        line_width = 0.75
        line_color = "#dddddd"
        terminal_state_border_color = "#0000ff"
        policy_color = "#ff0000"


        def draw_policy_for_state(ax, policy, state, color):
            """
            Helper function to draw a policy for a given state
            """

            def draw_arrow(ax, x, y, dir, color):
                """
                Helper function to draw an arrow
                """

                # Helper to conver to image y coordinates
                to_image_x = lambda xin: xin + 0.5
                to_image_y = lambda yin: self.height - (yin + 1) + 0.5

                # Render parameters
                head_width = 0.05
                head_length = 0.1
                length = 0.4 - head_length

                # Get start position
                start_x = to_image_x(x)
                start_y = to_image_y(y)

                # Compute length based on direction
                length_x = 0
                length_y = 0
                if dir == "N":
                    length_y = length
                elif dir == "NE":
                    length_x = length_y = length
                    length_x /= 1.4
                    length_y /= 1.4
                elif dir == "E":
                    length_x = length
                elif dir == "SE":
                    length_x = length
                    length_y = -length
                    length_x /= 1.4
                    length_y /= 1.4
                elif dir == "S":
                    length_y = -length
                elif dir == "SW":
                    length_x = length_y = -length
                    length_x /= 1.4
                    length_y /= 1.4
                elif dir == "W":
                    length_x = -length
                if dir == "NW":
                    length_x = -length
                    length_y = length
                    length_x /= 1.4
                    length_y /= 1.4

                ax.arrow(
                    start_x,
                    start_y,
                    length_x,
                    length_y,
                    head_width=head_width,
                    head_length=head_length,
                    facecolor=color,
                    edgecolor=color,
                    zorder=5
                )

            # Get the policy for this state
            sp = policy.policy_mapping[state]

            # Look up which actions we need to draw
            action_bv = np.array(list(sp.values())) != 0
            actions_to_draw = self.action_set[action_bv]

            for action in actions_to_draw:
                draw_arrow(ax, state.x, state.y, action, color)


        def draw_value_text(x, y, value, textcolor):
            """
            Helper function to plot the value of a state in text form
            """
            render_pos = (
                x + 0.5,
                self.height - (y + 1) + 0.5
            )

            # Draw state value as text
            plt.text(
                render_pos[0],
                render_pos[1],
                "{: .2f}".format(value),
                horizontalalignment="center",
                verticalalignment="center",
                family="serif",
                size=13,
                color=textcolor
            )


        # Helper lambda to invert a color
        high_contrast_color = lambda arr: [1 if c < 0.5 else 0 for c in arr]
        
        # Compute value scaling variables
        max_value = 1
        min_value = 0
        if value_function is not None:
            # Compute maxmium value so we can normalize
            for state in value_function:
                v = value_function[state]
                if v > max_value: max_value = v
                if v < min_value: min_value = v

        value_range = max_value - min_value
        value_to_color = lambda v: [(v - min_value) / value_range] * 3

        # Get the axes to draw to
        ax = plt.gca()

        # Terminal states are drawn before other states
        for state in self.terminal_state_set:

            value = max_value/2
            if value_function is not None:
                value = value_function[state]

            render_pos = (
                state.x,
                self.height - (state.y + 1)
            )
            render_color = value_to_color(value)

            ax.add_artist(plt.Rectangle(
                    render_pos,
                    width=1,
                    height=1,
                    facecolor=render_color,
                    edgecolor=terminal_state_border_color,
                    linewidth=10,
                )
            )

            if value_function is not None and show_value_text:
                draw_value_text(state.x, state.y, value_function[state], high_contrast_color(render_color))

            # Draw the state policy
            if policy is not None:
                draw_policy_for_state(
                    ax,
                    policy,
                    state,
                    policy_color
                )


        for yi in range(self.height):
            for xi in range(self.width):

                # Get state
                state = self.state_set[self.xy_to_index(xi, yi)]

                value = max_value/2
                if value_function is not None:
                    value = value_function[state]

                # pyplot is y-up, our internal representation is y-down
                # Therefore flip y axis
                render_pos = (
                    xi,
                    self.height - (yi + 1)
                )
                render_color = value_to_color(value)

                if state not in self.terminal_state_set:
                    ax.add_artist(plt.Rectangle(
                            render_pos,
                            width=1,
                            height=1,
                            color=render_color
                        )
                    )

                    if value_function is not None and show_value_text:
                        draw_value_text(state.x, state.y, value_function[state], high_contrast_color(render_color))

                    # Draw the state policy
                    if policy is not None:
                        draw_policy_for_state(
                            ax,
                            policy,
                            state,
                            policy_color
                        )



        # Draw horizontal grid lines
        for i in range(self.height - 1):
            ax.add_artist(plt.Line2D(
                    (0, self.width),
                    (i+1, i+1),
                    color=line_color,
                    linewidth=line_width
                )
            )

        # Draw vetical grid lines
        for i in range(self.width - 1):
            ax.add_artist(plt.Line2D(
                    (i+1, i+1),
                    (0, self.height),
                    color=line_color,
                    linewidth=line_width
                )
            )

        ax.set_aspect("equal", adjustable="box")
        plt.xlim([0, self.width])
        plt.ylim([0, self.height])

        ax.tick_params(length=0, labelbottom="off", labelleft="off")

        # Add title and subtitle
        plt.figtext(
            0.5125,
            0.925,
            "{}".format(title if title is not None else "GridWorld"),
            fontsize=14,
            ha='center'
        )
        plt.figtext(
            0.5125,
            0.89,
            "{}".format(subtitle if subtitle is not None else ""),
            fontsize=10,
            ha='center'
        )
        
        # Figure is now ready for display or saving


    def xy_to_index(self, x, y):
        """
        Helper method to convert a x, y 0-based indices to a linear 0-based index
        """
        return y * self.width + x


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


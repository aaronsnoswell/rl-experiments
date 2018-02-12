"""
An implementation of the Binary World Inverse Reinforcement Learning
benchmark introduced in Wulfmeier, M., Ondruska, P. & Posner, I.
Maximum Entropy Deep Inverse Reinforcement Learning. (2015).
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy_helpers import show_complete_array


class BinaryWorld:
    """
    An implementation of the Binary World Inverse Reinforcement
    Learning benchmark introduced in Wulfmeier, M., Ondruska,
    P. & Posner, I. Maximum Entropy Deep Inverse Reinforcement
    Learning. (2015).
    """

    StateColorBlueInt = 0
    StateColorBlueChar = 'b'
    StateColorRedInt = 1
    StateColorRedChar = 'r'

    # Minimum and maximum reward a state can have
    MinReward = -1
    MaxReward = 1


    def __init__(
            self,
            *,
            size=32,
            feature_window_size=3,
            positive_score_count=4,
            negative_score_count=5,
            state_grid=None,
            random_seed=1337,
            name=None
            ):
        """
        Constructor
        """

        assert (feature_window_size % 2) == 1, "Feature window size must be odd"

        assert positive_score_count != negative_score_count, \
            "Positive score count ({}) cannot be equal to negative score count".format(positive_score_count)

        # Copy parameters locally
        self.size = size
        self.feature_window_size = feature_window_size
        self.positive_score_count = positive_score_count
        self.negative_score_count = negative_score_count
        self.random_seed = random_seed
        self.name = name

        if state_grid is not None:
            # Use the passed state grid

            assert len(state_grid.shape) == 2, \
                "State Grid must be a two-dimensional ndarray"

            assert (state_grid.shape[0] == state_grid.shape[1]), \
                "State Grid must be square"

            print("Using supplied state grid")
            print("Size (if given) will be ignored")

            self.size = state_grid.shape[0]
            self.state_grid = state_grid

        else:
            # Initialize a randomized state grid
            np.random.seed(random_seed)
            self.state_grid = np.random.randint(
                BinaryWorld.StateColorBlueInt,
                high=BinaryWorld.StateColorRedInt + 1,
                size=[self.size, self.size]
            )

        # Initialize the features and rewards
        self._initialize_features_and_rewards()


    def _initialize_features_and_rewards(self):
        """
        Internal function - initializes the feature and reward grids
        """

        # Figure out the actual padding size
        pad_size = self.feature_window_size // 2

        # Pad the state grid so we can do feature lookups easily
        self.state_grid_padded = np.pad(
            self.state_grid,
            (pad_size, pad_size),
            'constant',
            constant_values=-1
        )

        # Initialize reward matrix
        self.reward_grid = np.empty(
            shape=(self.size, self.size),
            dtype=int
        )

        # Initialise feature vector grid
        self.feature_grid = np.empty(
            shape=(self.size, self.size, self.feature_window_size ** 2),
            dtype=int
        )

        # Loop over every point in the state grid
        for y in range(self.size):
            for x in range(self.size):

                # For each point, loop over the feature window to compute
                # the feature vector
                feature_index = 0
                for yi in range(-pad_size, pad_size+1):
                    y_index = pad_size + y + yi

                    for xi in range(-pad_size, pad_size+1):
                        x_index = pad_size + x + xi

                        self.feature_grid[y][x][feature_index] = self.state_grid_padded[y_index][x_index]

                        feature_index += 1

                # Then compute and store the reward
                self.reward_grid[y][x] = self.get_reward_from_feature_vector(self.feature_grid[y][x])


    def _human_friendly_array_string(self, object_in):
        """
        Replaces 0 or 1 (the internal state representation) with
        'b' or 'r' respectively
        """
        string_out = str(object_in).replace("0", BinaryWorld.StateColorBlueChar).replace("1", BinaryWorld.StateColorRedChar)
        return string_out


    def __str__(self):
        """
        Get human-friendly string version of class
        """
        with show_complete_array():
            return "BinaryWorld(\n  S = {},\n  V = {}\n)".format(
                str(self._human_friendly_array_string(self.state_grid)).replace("\n", "\n      "),
                str(self.reward_grid).replace("\n", "\n      ")
            )


    def get_reward_from_feature_vector(self, f):
        """
        Utility function to return the ground truth reward for a given feature vector
        """

        num_blue_neighbours = np.sum(np.equal(f, BinaryWorld.StateColorBlueInt))
        if num_blue_neighbours == self.positive_score_count:
            return +1
        elif num_blue_neighbours == self.negative_score_count:
            return -1
        else:
            return 0


    def get_feature_vector(self, x, y):
        """
        Gets the feature vector for a state (x, y)
        A feature vector value of -1 indicates neighbouring states sampled outside the world grid
        """

        return self.feature_grid[y][x]


    def get_reward(self, x, y):
        """
        Gets the reward of a state (x, y)
        """

        return self.reward_grid[y][x]


    def generate_figure(self):
        """
        Internal method - generates a figure for display or saving
        """

        line_width = 0.75
        point_radius = 0.325
        line_color = "#efefef"

        fig = plt.figure()
        ax = plt.gca()

        # Plot rewards
        remap_range = lambda r: float(r - BinaryWorld.MinReward) / (BinaryWorld.MaxReward - BinaryWorld.MinReward)
        reward_colors=np.array([
            [remap_range(a), remap_range(a), remap_range(a)] for a in np.ravel(
                np.flip(self.reward_grid, 0)
            )
        ])
        reward_points = np.column_stack(
            np.where(
                np.flip(self.reward_grid, 0) != np.nan
            )
        )
        for i in range(len(reward_points)):
            pt = reward_points[i]
            c = reward_colors[i]
            ax.add_artist(
                plt.Rectangle(
                    (pt[1], pt[0]),
                    width=1,
                    height=1,
                    color=c)
                )
        
        # Plot blue states
        for pt in np.column_stack(np.where(np.flip(self.state_grid, 0) == BinaryWorld.StateColorBlueInt)):
            ax.add_artist(
                plt.Circle(
                    (pt[1] + 0.5, pt[0] + 0.5),
                    radius=point_radius,
                    color='blue')
                )

        # Plot red states
        for pt in np.column_stack(np.where(np.flip(self.state_grid, 0) == BinaryWorld.StateColorRedInt)):
            ax.add_artist(
                plt.Circle(
                    (pt[1] + 0.5, pt[0] + 0.5),
                    radius=point_radius,
                    color='red')
                )
        
        # Draw horizontal grid lines
        for i in range(self.size - 1):
            ax.add_artist(plt.Line2D(
                    (0, self.size),
                    (i+1, i+1),
                    color=line_color,
                    linewidth=line_width
                )
            )

        # Draw vetical grid lines
        for i in range(self.size - 1):
            ax.add_artist(plt.Line2D(
                    (i+1, i+1),
                    (0, self.size),
                    color=line_color,
                    linewidth=line_width
                )
            )


        ax.set_aspect("equal", adjustable="box")
        plt.xlim([0, self.size])
        plt.ylim([0, self.size])

        ax.tick_params(length=0, labelbottom="off", labelleft="off")

        # Add title
        plt.figtext(
            0.5125,
            0.925,
            "{}BinaryWorld".format(
                "{} ".format(self.name) if self.name is not None else ""
            ),
            fontsize=14,
            ha='center'
        )

        # Figure is now ready for display or saving
        return fig


    @staticmethod
    def save_figure(figure, filename, *, dpi=None):
        """
        Renders the given figure to an image
        """
        figure.savefig(
            filename,
            dpi=dpi,
            transparent=True,
            bbox_inches='tight',
            pad_inches=0
        )


def test_binaryworld():
    """
    Tests the BinaryWorld class
    """

    import pickle

    # Create a new random BinaryWorld
    bw = BinaryWorld(size=13)
    print(bw)

    # Test saving and loading to/from pickles
    filename = "sample_binaryworld.pickle"
    with open(filename, "wb") as file:          
        print("Saving to {}".format(filename))
        pickle.dump(bw, file)
        print("Done")

    with open(filename, "rb") as file:          
        print("Loading from {}".format(filename))
        bw_loaded = pickle.load(file)
        assert (not np.any(bw_loaded.state_grid != bw.state_grid)), "Loaded state grid did not match saved"
        print("Done")

    # Test save and display functionality
    fig = bw.generate_figure()
    plt.show()


if __name__ == "__main__":
    test_binaryworld()

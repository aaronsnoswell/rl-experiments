"""
An implementation of the Binary World Inverse Reinforcement Learning
benchmark introduced in Wulfmeier, M., Ondruska, P. & Posner, I.
Maximum Entropy Deep Inverse Reinforcement Learning. (2015).
"""


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from contextlib import contextmanager

@contextmanager
def show_complete_array():
    """
    Shows a complete numpy array
    From https://stackoverflow.com/a/45831462/885287
    """
    oldoptions = np.get_printoptions()
    np.set_printoptions(threshold=np.inf)
    yield
    np.set_printoptions(**oldoptions)


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

    # Minimum and maximum value a state can have
    MinValue = -1
    MaxValue = 1


    def __init__(
            self,
            *,
            width=32,
            height=32,
            feature_window_size_x=3,
            feature_window_size_y=3,
            positive_score_count=4,
            negative_score_count=5,
            state_grid=None,
            random_seed=1337
            ):
        """
        Constructor
        """

        assert (feature_window_size_x % 2) == 1, "Feature window dimensions must be odd"
        assert (feature_window_size_y % 2) == 1, "Feature window dimensions must be odd"

        assert positive_score_count != negative_score_count, \
            "Positive score count ({}) cannot be equal to negative score count".format(positive_score_count)

        # Copy parameters locally
        self.width = width
        self.height = height
        self.feature_window_size_x = feature_window_size_x
        self.feature_window_size_y = feature_window_size_y
        self.positive_score_count = positive_score_count
        self.negative_score_count = negative_score_count
        self.random_seed = random_seed

        if state_grid is not None:
            # Use the passed state grid

            assert len(state_grid.shape) == 2, \
                "State Grid must be a two-dimensional ndarray"

            print("Using supplied state grid")
            print("Width and height values (if given) will be ignored")

            self.height, self.width = state_grid.shape
            self.state_grid = state_grid

        else:
            # Initialize a randomized state grid
            np.random.seed(random_seed)
            self.state_grid = np.random.randint(
                BinaryWorld.StateColorBlueInt,
                high=BinaryWorld.StateColorRedInt + 1,
                size=[self.height, self.width]
            )

        # Initialize the features and values
        self._initialize_features_and_values()


    def _initialize_features_and_values(self):
        """
        Internal function - initializes the feature and value grids
        """

        # Figure out the actual padding size
        pad_x = self.feature_window_size_x // 2
        pad_y = self.feature_window_size_y // 2

        # Pad the state grid so we can do feature lookups easily
        self.state_grid_padded = np.pad(
            self.state_grid,
            (pad_x, pad_y),
            'constant',
            constant_values=-1
        )

        # Initialize value matrix
        self.value_grid = np.empty(
            shape=(self.height, self.width),
            dtype=int
        )

        # Initialise feature vector grid
        self.feature_grid = np.empty(
            shape=(self.height, self.width, self.feature_window_size_x * self.feature_window_size_y),
            dtype=int
        )

        # Loop over every point in the state grid
        for y in range(self.height):
            for x in range(self.width):

                # For each point, loop over the feature window to compute
                # the feature vector
                feature_index = 0
                for yi in range(-pad_y, pad_y+1):
                    y_index = pad_y + y + yi

                    for xi in range(-pad_x, pad_x+1):
                        x_index = pad_x + x + xi

                        self.feature_grid[y][x][feature_index] = self.state_grid_padded[y_index][x_index]

                        feature_index += 1

                # Then compute and store the value
                self.value_grid[y][x] = self.get_reward_from_feature_vector(self.feature_grid[y][x])


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
                str(self.value_grid).replace("\n", "\n      ")
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


    def get_value(self, x, y):
        """
        Gets the value of a state (x, y)
        """

        return self.value_grid[y][x]


    def display(self):
        """
        Renders the current world state to an image
        """

        line_width = 3
        point_radius = 0.325

        fig = plt.figure()
        ax = plt.gca()

        # Plot values
        remap_range = lambda v: float(v - BinaryWorld.MinValue) / (BinaryWorld.MaxValue - BinaryWorld.MinValue)
        value_colors=np.array([
            [remap_range(a), remap_range(a), remap_range(a)] for a in np.ravel(
                np.flip(self.value_grid, 0)
            )
        ])
        value_points = np.column_stack(
            np.where(
                np.flip(self.value_grid, 0) != np.nan
            )
        )
        for i in range(len(value_points)):
            pt = value_points[i]
            c = value_colors[i]
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

        ax.set_aspect("equal", adjustable="box")
        plt.xlim([0, self.width])
        plt.ylim([0, self.height])

        #[i.set_linewidth(line_width) for i in ax.spines.values()]
        #plt.grid(True, color="black", linewidth=line_width)
        ax.tick_params(length=0, labelbottom="off", labelleft="off") 

        plt.show()


def test_binaryworld():
    """
    Tests the BinaryWorld class
    """

    import pickle

    # Create a new random BinaryWorld
    bw = BinaryWorld(width=13, height=13)
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

    # Test display functionality
    bw.display()


if __name__ == "__main__":
    test_binaryworld()

"""
An implementation of the Binary World Inverse Reinforcement Learning
benchmark introduced in Wulfmeier, M., Ondruska, P. & Posner, I.
Maximum Entropy Deep Inverse Reinforcement Learning. (2015).
"""


import numpy as np
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


    def __init__(self, *, width=32, height=32, feature_window_size_x=3, feature_window_size_y=3, positive_score_count=4, negative_score_count=5):
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
        
        # Initialize a randomized state grid
        self.state_grid = np.random.randint(
            BinaryWorld.StateColorBlueInt,
            high=BinaryWorld.StateColorRedInt + 1,
            size=[height, width]
        )

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
            shape=(height, width),
            dtype=int
        )

        # Initialise feature vector grid
        self.feature_grid = np.empty(
            shape=(height, width, self.feature_window_size_x * self.feature_window_size_y),
            dtype=int
        )

        # Loop over every point in the state grid
        for y in range(height):
            for x in range(width):

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
        # TODO see https://matplotlib.org/gallery/lines_bars_and_markers/simple_plot.html
        pass


def test_binaryworld():
    """
    Tests the BinaryWorld class
    """

    a = BinaryWorld(width=10, height=10)
    print(a)


if __name__ == "__main__":
    test_binaryworld()

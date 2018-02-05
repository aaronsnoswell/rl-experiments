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


    def __init__(self, *, width=32, height=32, window_size_x=3, window_size_y=3):
        """
        Constructor
        """

        assert (window_size_x % 2) == 1, "Window dimensions must be odd"
        assert (window_size_y % 2) == 1, "Window dimensions must be odd"
        
        # Initialize a randomized state grid
        self.state_grid = np.random.randint(
            BinaryWorld.StateColorBlueInt,
            high=BinaryWorld.StateColorRedInt + 1,
            size=[height, width]
        )

        # Figure out the actual padding size
        pad_x = window_size_x // 2
        pad_y = window_size_y // 2

        # Pad the state grid so we can do feature lookups easily
        self.state_grid_padded = np.pad(
            self.state_grid,
            (pad_x, pad_y),
            'constant',
            constant_values=-1
        )

        # Initialise and populate the feature vector grid
        self.feature_vectors = np.empty(
            shape=(height, width, window_size_x * window_size_y),
            dtype=int
        )

        # Loop over every point in the state grid
        for y in range(height):
            for x in range(width):

                # For each point, loop over the feature window
                feature_index = 0
                for yi in range(-pad_y, pad_y+1):
                    y_index = pad_y + y + yi

                    for xi in range(-pad_x, pad_x+1):
                        x_index = pad_x + x + xi

                        # Compute and store the feature vector
                        self.feature_vectors[y][x][feature_index] = self.state_grid_padded[y_index][x_index]

                        feature_index += 1


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
            return self._human_friendly_array_string(
                "BinaryWorld(\n{}\n)".format(
                    "  " + str(self.state_grid).replace("\n", "\n  ")
                )
            )


    def get_reward_from_feature_vector(self, f, *, positive_score_count=4, negative_score_count=5):
        """
        Returns the ground truth reward for a given feature vector
        """

        assert positive_score_count != negative_score_count, "Positive score count ({}) cannot be equal to negative score count".format(positive_score_count)

        num_blue_neighbours = np.sum(np.equal(f, BinaryWorld.StateColorBlueInt))
        if num_blue_neighbours == positive_score_count:
            return +1
        elif num_blue_neighbours == negative_score_count:
            return -1
        else:
            return 0


    def get_feature_vector(self, x, y):
        """
        Gets the feature vector for a state (x, y)
        A value of -1 indicates neighbouring states sampled outside the world grid
        """

        return self.feature_vectors[y][x]


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
    f = a.get_feature_vector(3, 3)
    print(a._human_friendly_array_string(f))
    print(a.get_reward_from_feature_vector(f))


if __name__ == "__main__":
    test_binaryworld()

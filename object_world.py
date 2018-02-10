"""
An implementation of the Object World Inverse Reinforcement Learning
benchmark introduced in Levine, S., Popovic, Z., and Koltun, V.
Nonlinear inverse reinforcement learning with gaussian processes. (2011).
"""


import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy_helpers import show_complete_array


class ObjectWorld:
    """
    An implementation of the Object World Inverse Reinforcement Learning
    benchmark introduced in Levine, S., Popovic, Z., and Koltun, V.
    Nonlinear inverse reinforcement learning with gaussian processes. (2011).
    """

    # Helpful constants
    OuterColorIndex = 0
    InnerColorIndex = 1

    # Indices if the first and second colors
    # (used for determining the reward, hence they are special)
    Color1Index = 0
    Color2Index = 1
    
    # If a state has no object, it's colors will be set to this value
    NoObjectColorValue = -1

    # Minimum and maximum reward a state can have
    MinReward = -1
    MaxReward = 1


    def __init__(
            self,
            *,
            size=32,
            num_colors=7,
            object_likelihood=0.2,
            use_continuous_features=True,
            state_grid=None,
            random_seed=1337
            ):
        """
        Constructor
        """

        # Copy parameters locally
        self.size = size
        self.num_colors = num_colors
        self.object_likelihood = object_likelihood
        self.use_continuous_features = use_continuous_features
        self.random_seed = random_seed

        supported_colors = len(self._get_color_list())
        assert (self.num_colors <= supported_colors), \
            "Currently only supports up to {} colors".format(supported_colors)

        if state_grid is not None:
            # Use the passed state grid

            assert len(state_grid.shape) == 3, \
                "State Grid must be a three-dimensional ndarray"

            assert (state_grid.shape[0] == state_grid.shape[1]), \
                "State Grid dimensions 0 and 1 must be equal"

            assert (state_grid.shape[2] == 2), \
                "3rd dimension of state grid must be of size 2 (i.e. outer and inner color)"

            print("Using supplied state grid")
            print("size, num_colors and object_likelihood (if given) will be ignored")

            self.state_grid = state_grid

            # Ensure internal state is consistent
            self.size = self.state_grid.shape[0]
            self.num_colors = np.sum(np.unique(self.state_grid) > -1)
            self.object_likelihood = np.sum(self.state_grid[:, :, 0] > -1) / self.size ** 2
            self.random_seed = None


        else:
            # Initialize a randomized state grid
            np.random.seed(random_seed)

            self.state_grid = np.empty(
                shape=(self.size, self.size, 2),
                dtype=int
            )

            num_non_object_options = round(num_colors / self.object_likelihood - num_colors)
            options = list(range(num_colors)) + num_non_object_options * [self.NoObjectColorValue]

            for yi in range(self.size):
                for xi in range(self.size):
                    ii = np.random.choice(len(options))
                    outer_color = options[ii]

                    inner_color = self.NoObjectColorValue
                    if outer_color != self.NoObjectColorValue:
                        inner_color = np.random.choice(num_colors)

                    self.state_grid[yi][xi][self.OuterColorIndex] = outer_color
                    self.state_grid[yi][xi][self.InnerColorIndex] = inner_color

        # Initialize the features and rewards
        self._initialize_features_and_rewards()


    def _initialize_features_and_rewards(self):
        """
        Internal function - initializes the feature and reward grids
        """

        # Cts. feature representation dimensions are as follows:
        # y, x, color position (ourside, then in), color
        self._cf = np.empty(
            shape=(self.size, self.size, 2, self.num_colors),
            dtype=float
        )

        # Each discrete feature dimension indicates if an object of that
        # color position of that color is closer than d \member {1, ..., self.size}
        self._df = np.empty(
            shape=(self.size, self.size, 2, self.num_colors, self.size),
            dtype=bool
        )

        # Initialise the reward grid
        self.reward_grid = np.empty(
            shape=(self.size, self.size),
            dtype=int
        )

        # Compute features
        for yi in range(self.size):
            for xi in range(self.size):

                # Compute features
                # For each color
                for cii in range(self.num_colors):

                    closest_outer_color_dist = math.inf
                    closest_inner_color_dist = math.inf
                     
                    # Loop over the whole grid
                    for yii in range(self.size):
                        for xii in range(self.size):

                            dist = math.sqrt((yii - yi) ** 2 + (xii - xi) ** 2)

                            # Checking for the closest object with that outer color...
                            if self.state_grid[yii, xii, self.OuterColorIndex] == cii:
                                if dist < closest_outer_color_dist:
                                    closest_outer_color_dist = dist

                            # And checking for the closest object with that inner color
                            if self.state_grid[yii, xii, self.InnerColorIndex] == cii:
                                if dist < closest_inner_color_dist:
                                    closest_inner_color_dist = dist

                    # Finally, store the closest inner/outer color distances
                    self._cf[yi, xi, self.OuterColorIndex, cii] = closest_outer_color_dist
                    self._cf[yi, xi, self.InnerColorIndex, cii] = closest_inner_color_dist

                    # And a binary vector indicating distance to the nearest object of that color
                    # up to a max distance of self.size
                    for n in range(1, self.size + 1):
                        _cf = self._cf[yi, xi, self.OuterColorIndex, cii]
                        self._df[yi, xi, self.OuterColorIndex, cii, n-1] = _cf < n
                        self._df[yi, xi, self.InnerColorIndex, cii, n-1] = _cf < n
                # End Compute Features

        # Compute rewards
        positive = []
        negative = []

        if self.use_continuous_features:
            mindist_outer_color_1 = self._cf[:, :, ObjectWorld.OuterColorIndex, ObjectWorld.Color1Index]
            mindist_outer_color_2 = self._cf[:, :, ObjectWorld.OuterColorIndex, ObjectWorld.Color2Index]

            color_1_within_3 = mindist_outer_color_1 <= 3
            color_2_within_2 = mindist_outer_color_2 <= 2

            positive = np.logical_and(color_1_within_3, color_2_within_2)
            negative = np.logical_and(color_1_within_3, np.logical_not(color_2_within_2))
        else:
            bv_color_1 = self._df[:, :, ObjectWorld.OuterColorIndex, ObjectWorld.Color1Index]
            bv_color_2 = self._df[:, :, ObjectWorld.OuterColorIndex, ObjectWorld.Color2Index]
            
            color_1_within_3 = np.all(bv_color_1[:, :, 3-1:], axis=2)
            color_2_within_2 = np.all(bv_color_2[:, :, 2-1:], axis=2)

            positive = np.logical_and(color_1_within_3, color_2_within_2)
            negative = np.logical_and(color_1_within_3, np.logical_not(color_2_within_2))

        for yi in range(self.size):
            for xi in range(self.size):
                if positive[yi, xi]:
                    self.reward_grid[yi, xi] = 1
                elif negative[yi, xi]:
                    self.reward_grid[yi, xi] = -1
                else:
                    self.reward_grid[yi, xi] = 0


    def __str__(self):
        """
        Get human-friendly string version of class
        """
        with show_complete_array():
            return "ObjectWorld(\n  S = {},\n  V = {}\n)".format(
                str(self.state_grid).replace("\n", "\n      "),
                str(self.reward_grid).replace("\n", "\n      ")
            )


    def get_reward_from_feature_vector(self, f):
        """
        Utility function to return the ground truth reward for a given feature vector

        State reward is determined as follows:
        The reward is positive for cells which are both within the
        distance 3 of outer color 1 and distance 2 of outer color 2,
        negative if only within distance 3 of outer color 1 and zero
        otherwise. Inner colors are distractors.
        """

        if self.use_continuous_features:

            if f[self.OuterColorIndex, self.Color1Index] <= 3:
                if f[self.OuterColorIndex, self.Color2Index] <= 2:
                    return 1
                else:
                    return -1
            else:
                return 0

        else:

            outer_color_1 = f[self.OuterColorIndex, self.Color1Index]
            outer_color_2 = f[self.OuterColorIndex, self.Color2Index]
            if (not np.any(outer_color_1[0:3-1])) and np.all(outer_color_1[3-1:]):
                if (not np.any(outer_color_2[0:2-1])) and np.all(outer_color_2[2-1:]):
                    return 1
                else:
                    return -1
            else:
                return 0


    def get_feature_vector(self, x, y):
        """
        Gets the feature vector for a state (x, y)
        """
        if self.use_continuous_features:
            return self._cf[y, x]
        else:
            return self._df[y, x]


    def get_reward(self, x, y):
        """
        Gets the reward of a state (x, y)
        """
        return self.reward_grid[y, x]


    def _get_color_list(self):
        """
        Returns a color list
        """
        # These are the "Tableau 20" colors as RGB.  
        tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),  
                     (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),  
                     (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),  
                     (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),  
                     (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]  
          
        # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.  
        for i in range(len(tableau20)):  
            r, g, b = tableau20[i]  
            tableau20[i] = (r / 255., g / 255., b / 255.)

        return tableau20


    def _generate_figure(self, *, show_inner_colors=True, color_array=None):
        """
        Internal method - generates a figure for display or saving
        """

        line_width = 0.75
        point_radius = 0.325
        line_color = "#efefef"

        fig = plt.figure()
        ax = plt.gca()

        # Plot rewards
        remap_range = lambda r: float(r - ObjectWorld.MinReward) / (ObjectWorld.MaxReward - ObjectWorld.MinReward)
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
        

        # Plot objects
        color_list = color_array
        if color_list is None:
            color_list = self._get_color_list()
        for yi in range(self.size):
            for xi in range(self.size):
                state = np.flip(self.state_grid, 0)[yi][xi]

                if state[0] == -1:
                    continue

                ax.add_artist(plt.Circle(
                        (xi + 0.5, yi + 0.5),
                        radius=point_radius,
                        color=color_list[state[0]]
                    )
                )

                if show_inner_colors:
                    ax.add_artist(plt.Circle(
                            (xi + 0.5, yi + 0.5),
                            radius=point_radius * 0.4,
                            color=color_list[state[1]]
                        )
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

        # Figure is now ready for display or saving
        return fig


    def display_figure(self, *, show_inner_colors=True, color_array=None):
        """
        Renders the current world state to an image
        """
        fig = self._generate_figure(show_inner_colors=show_inner_colors, color_array=color_array)
        plt.show()


    def save_figure(self, filename, *, show_inner_colors=True, color_array=None, dpi=None):
        """
        Renders the current world state to an image
        """
        fig = self._generate_figure(show_inner_colors=show_inner_colors, color_array=color_array)
        fig.savefig(
            filename,
            dpi=dpi,
            transparent=True,
            bbox_inches='tight',
            pad_inches=0
        )
        plt.close()


def test_objectworld():
    """
    Tests the ObjectWorld class
    """

    import pickle

    # Create a new random ObjectWorld
    ow = ObjectWorld(size=13)  
    print(ow)

    # Test saving and loading to/from pickles
    filename = "sample_objectworld.pickle"
    with open(filename, "wb") as file:          
        print("Saving to {}".format(filename))
        pickle.dump(ow, file)
        print("Done")

    with open(filename, "rb") as file:          
        print("Loading from {}".format(filename))
        ow_loaded = pickle.load(file)
        assert (not np.any(ow_loaded.state_grid != ow.state_grid)), "Loaded state grid did not match saved"
        print("Done")

    # Test save and display functionality
    ow.save_figure("sample_objectworld.pdf")
    ow.display_figure()


if __name__ == "__main__":
    test_objectworld()

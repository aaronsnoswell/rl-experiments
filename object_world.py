"""
An implementation of the Object World Inverse Reinforcement Learning
benchmark introduced in Levine, S., Popovic, Z., and Koltun, V.
Nonlinear inverse reinforcement learning with gaussian processes. (2011).
"""

import math
import numpy as np
import matplotlib.pyplot as plt

from colormaps import tableau20
from numpy_helpers import show_complete_array


class ObjectWorld():
    """
    An implementation of the Object World Inverse Reinforcement Learning
    benchmark introduced in Levine, S., Popovic, Z., and Koltun, V.
    Nonlinear inverse reinforcement learning with gaussian processes. (2011).
    """

    # Minimum and maximum reward a state can have
    MinReward = -1
    MaxReward = 1

    # Distance required to be close to color 0 and color 1
    Color1Dist = 3
    Color2Dist = 2


    class Color():
        # Simple color class

        def __init__(self, value):
            # Constructor
            self.value = value

        def __str__(self):
            # Get string repr
            return "ObjectWorld.Color(value={})".format(
                self.value
            )

        def __repr__(self):
            # Get machine repr
            return str(self.value)

        def __lt__(self, other):
            # Less than comparator
            return self.value < other.value

        def __eq__(self, other):
            # Equality comparator
            if isinstance(other, self.__class__):
                return self.__dict__ == other.__dict__
            else:
                return False


    class Object():
        # Simple object class

        def __init__(self, outer_color, inner_color):
            # Constructor
            self.outer_color = outer_color
            self.inner_color = inner_color

        def __str__(self):
            # Get string repr
            return "ObjectWorld.Object(outer_color={}, inner_color={})".format(
                self.outer_color.__repr__(),
                self.inner_color.__repr__()
            )

        def __repr__(self):
            # Get machine repr
            return "({},{})".format(
                self.outer_color.__repr__(),
                self.inner_color.__repr__()
            )

        def __lt__(self, other):
            # Less than comparator
            return self.__repr__() < other.__repr__()

        def __eq__(self, other):
            # Equality comparator
            if isinstance(other, self.__class__):
                return (self.outer_color == other.outer_color) \
                    and (self.inner_color == other.inner_color)
            else:
                return False


    def __init__(
        self,
        colors,
        *,
        size=32,
        object_likelihood=0.2,
        use_continuous_features=True,
        use_manhattan_distance=False,
        wrap_at_edges=True,
        state_grid=None,
        name=None,
        random_seed=None
    ):
        """
        Constructor
        """

        assert colors is not None, \
            "Color array must be provided"

        # Copy parameters locally
        self.colors = colors
        self.size = size
        self.object_likelihood = object_likelihood
        self.use_continuous_features = use_continuous_features
        self.use_manhattan_distance = use_manhattan_distance
        self.wrap_at_edges = wrap_at_edges
        self.random_seed = random_seed
        self.name = name

        if state_grid is not None:

            assert len(state_grid.shape) == 2, \
                "State Grid must be a two-dimensional ndarray"

            assert (state_grid.shape[0] == state_grid.shape[1]), \
                "State Grid must be square"

            # Check for consistency
            supplied_colors = np.unique(
                np.array(
                    sum(
                        [[o.inner_color, o.outer_color] for o in state_grid[state_grid != None]],
                        []
                    )
                )
            )
            for c in supplied_colors:
                assert c in self.colors, \
                    "Color {} was not in supplied color array".format(c)

            # Use the supplied state grid
            print("Using given state grid")
            print("Size, object likelihood and random seed values (if given) will be ignored")

            self.state_grid = state_grid

            # Ensure consistent internal state
            self.size = self.state_grid.shape[0]
            self.object_likelihood = np.sum(self.state_grid != None) / self.size ** 2
            self.random_seed = None

        else:

            # Compute a new random ObjectWorld
            np.random.seed(random_seed)
            self.state_grid = np.array([[None] * self.size for i in range(self.size)])
            for yi in range(self.size):
                for xi in range(self.size):
                    if np.random.random() < self.object_likelihood:
                        ob = ObjectWorld.Object(
                            self.colors[np.random.choice(len(self.colors))],
                            self.colors[np.random.choice(len(self.colors))]
                        )
                        self.state_grid[yi, xi] = ob

        # Initialize feature and reward grids
        if use_continuous_features:
            self.feature_grid = np.empty(
                shape=(self.size, self.size, len(self.colors) * 2),
                dtype=float
            )
        else:
            self.feature_grid = np.empty(
                shape=(self.size, self.size, len(self.colors) * 2 * self.size)
            )

        self.reward_grid = np.empty(
            shape=(self.size, self.size),
            dtype=int
        )

        for yi in range(self.size):
            for xi in range(self.size):
                f = self.get_feature_for_state(xi, yi)
                self.feature_grid[yi, xi] = f
                self.reward_grid[yi, xi] = self.get_reward_from_feature(f)


    def get_feature_for_state(self, x, y):
        """
        Compute the feature vector for a given state (x, y)
        """

        # A padding term is used to account for an infinite grid (wraps at edges)
        padding = 0
        if self.wrap_at_edges:
            padding = max(ObjectWorld.Color1Dist, ObjectWorld.Color2Dist)

        # There are 2C continuous features - one for the euclidian
        # distance to the nearest object of outer, and inner color
        f = np.zeros(shape=len(self.colors) * 2)

        for c in range(len(self.colors)):

            smallest_outer_dist = math.inf
            smallest_inner_dist = math.inf

            for yi in range(-padding, self.size + padding):
                for xi in range(-padding, self.size + padding):

                    state = self.state_grid[yi % self.size, xi % self.size]
                    if state is None: continue

                    dist = math.sqrt((yi-y) ** 2 + (xi-x) ** 2)
                    if self.use_manhattan_distance:
                        dist = np.abs(yi - y) + np.abs(xi - x)

                    # Outer color check
                    if state.outer_color == self.colors[c]:
                        if dist < smallest_outer_dist:
                            smallest_outer_dist = dist

                    # Inner color check
                    if state.inner_color == self.colors[c]:
                        if dist < smallest_inner_dist:
                            smallest_inner_dist = dist

            f[c] = smallest_outer_dist
            f[len(self.colors) + c] = smallest_inner_dist

        if self.use_continuous_features:
            return f
        else:
            f_discrete = np.array(
                [False for i in range(len(self.colors) * 2 * self.size)],
                dtype=bool
            )

            for c in range(len(self.colors)):
                for n in range(1, self.size + 1):
                    # Store outer color binary map
                    f_discrete[c * self.size + (n - 1)] = f[c] < n

                    # Store inner color binary map
                    f_discrete[len(self.colors) * self.size + c * self.size + (n - 1)] = f[len(self.colors) + c] < n

            return f_discrete


    def get_reward_from_feature(self, f):
        """
        Compute the reward vector for a given feature f
        """

        if self.use_continuous_features:

            if f[0] <= ObjectWorld.Color1Dist:
                # Close to color 0
                if f[1] <= ObjectWorld.Color2Dist:
                    # Close to color 1
                    return 1
                else:
                    return -1
            else:
                return 0

        else:

            color_0_outer_binary_map = f[0:self.size]
            color_1_outer_binary_map = f[self.size:(self.size * 2)]
            #color_0_inner_binary_map = f[(self.size * 2):(self.size*3)]
            #color_1_inner_binary_map = f[(self.size * 3):]

            if np.all(color_0_outer_binary_map[ObjectWorld.Color1Dist-1:]):
                # Close to color 0
                if np.all(color_1_outer_binary_map[ObjectWorld.Color2Dist-1:]):
                    # Close to color 1
                    return 1
                else:
                    return -1
            else:
                return 0


    def __str__(self):
        # Get string repr
        with show_complete_array():
            return "ObjectWorld(\n  S={},\n  R={}\n)".format(
                str(self.state_grid).replace("\n", "\n    "),
                str(self.reward_grid).replace("\n", "\n    ")
            )


    def get_primary_color(self):
        return self.colors[0]


    def get_secondary_color(self):
        return self.colors[1]


    def get_relevant_objects(self, *, unique=True):
        arr = np.array([])
        for yi in range(self.size):
            for xi in range(self.size):
                state = self.state_grid[yi, xi]
                if state is None: continue
                if state.outer_color in self.colors[0:1]:
                    arr = np.append(state, arr)

        if unique:
            return np.unique(arr)
        else:
            return arr


    def get_distractor_objects(self, *, unique=True):
        arr = np.array([])
        for yi in range(self.size):
            for xi in range(self.size):
                state = self.state_grid[yi, xi]
                if state is None: continue
                if state.outer_color not in self.colors[0:1]:
                    arr = np.append(state, arr)

        if unique:
            return np.unique(arr)
        else:
            return arr


    def generate_figure(self, show_distractors=True):
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
            [remap_range(r), remap_range(r), remap_range(r)] for r in np.ravel(
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
        for yi in range(self.size):
            for xi in range(self.size):
                state = np.flip(self.state_grid, 0)[yi, xi]
                if state is None: continue

                if state.outer_color not in self.colors[0:2]:
                    # This object is a distractor
                    if not show_distractors:
                        continue

                ax.add_artist(plt.Circle(
                        (xi + 0.5, yi + 0.5),
                        radius=point_radius,
                        color=state.outer_color.value
                    )
                )

                ax.add_artist(plt.Circle(
                        (xi + 0.5, yi + 0.5),
                        radius=point_radius * 0.4,
                        color=state.inner_color.value
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

        # Add title and subtitle
        plt.figtext(
            0.5125,
            0.925,
            "{}ObjectWorld".format(
                "{} ".format(self.name) if self.name is not None else ""
            ),
            fontsize=14,
            ha='center'
        )
        plt.figtext(
            0.5125,
            0.89,
            "({}, {}, {})".format(
                "Continuous" if self.use_continuous_features else "Discrete",
                "Manhattan" if self.use_manhattan_distance else "Euclidian",
                "Infinite" if self.wrap_at_edges else "Finite",
            ),
            fontsize=10,
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



def test_objectworld():
    """
    Tests the ObjectWorld class
    """

    import pickle

    seed = np.random.randint(5000)
    print("Using seed {}".format(seed))

    colors = [ObjectWorld.Color(c) for c in tableau20[0:7]]
    #colors = [ObjectWorld.Color('r'), ObjectWorld.Color('g')]
    size=13

    # Create a new random ObjectWorld
    ow = ObjectWorld(
        colors=colors,
        size=size,
        wrap_at_edges=True,
        random_seed=seed
    )
    print(ow)
    #print(ow.feature_grid)
    #print(ow.get_primary_color())
    #print(ow.get_secondary_color())
    #print(ow.get_relevant_objects())
    #print(ow.get_distractor_objects())

    # Compare with the discrete version
    #ow_disc = ObjectWorld(
    #    colors=colors,
    #    size=size,
    #    use_continuous_features=False,
    #    random_seed=seed
    #)
    #fig = ow_disc.generate_figure()
    #plt.ion()
    #plt.show()
    #plt.ioff()

    # Test saving and loading to/from pickles
    filename = "sample_objectworld.pickle"
    with open(filename, "wb") as file:          
        print("Saving to {}".format(filename))
        pickle.dump(ow, file)
        print("Done")

    with open(filename, "rb") as file:          
        print("Loading from {}".format(filename))
        ow_loaded = pickle.load(file)
        assert (not np.any(ow_loaded.state_grid != ow.state_grid)), \
            "Loaded state grid did not match saved"
        print("Done")

    # Test save and display functionality
    fig = ow.generate_figure()
    ObjectWorld.save_figure(fig, "sample_objectworld.pdf")
    plt.show()


if __name__ == "__main__":
    test_objectworld()

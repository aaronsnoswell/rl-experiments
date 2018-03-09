
import numpy as np
import matplotlib.pyplot as plt
from contextlib import contextmanager

@contextmanager
def showall():
    """
    Helper to show all elements in a numpy array when using print() e.g.
    with showall():
        print(my_array)
    """
    oldoptions = np.get_printoptions()
    np.set_printoptions(threshold=np.inf)
    yield
    np.set_printoptions(**oldoptions)



def high_contrast_color(color):
    """
    Inverts a 3-tuple pyplot color to get a high contrast alternative color
    """
    return [1 if c < 0.5 else 0 for c in color]


def draw_text(x, y, height, value, *, textcolor="black", formatstr="{: .2f}"):
    """
    Helper function to plot text at an x, y location
    """
    render_pos = (
        x + 0.5,
        height - (y + 1) + 0.5
    )

    # Draw state value as text
    plt.text(
        render_pos[0],
        render_pos[1],
        formatstr.format(value),
        horizontalalignment="center",
        verticalalignment="center",
        family="serif",
        size=13,
        color=textcolor
    )

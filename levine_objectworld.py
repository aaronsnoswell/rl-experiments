
import numpy as np
import matplotlib.pyplot as plt

from object_world import ObjectWorld


# Define some colors
blue = ObjectWorld.Color('#0000FF')
green = ObjectWorld.Color('#008000')
red = ObjectWorld.Color('#FF0000')
cyan = ObjectWorld.Color('#00BFBF')
pink = ObjectWorld.Color('#BF00BF')
yellow = ObjectWorld.Color('#BFBF00')
grey = ObjectWorld.Color('#404040')
colors = [blue, green, red, cyan, pink, yellow, grey]

"""
NB: The below code re-creates the visible portion of the ObjectWorld diagram
shown on page 6 of Levine, S., Popovic, Z., and Koltun, V. Nonlinear inverse
reinforcement learning with gaussian processes. (2011).

Because the diagram on that page is a sub-set of the true ObjectWorld state,
the edges of the state diagram won't match
"""
size = 16
state_grid = np.array([[None] * size for i in range(size)])

state_grid[0,   0] = ObjectWorld.Object(yellow, red)
state_grid[0,   5] = ObjectWorld.Object(red,    cyan)
state_grid[0,  11] = ObjectWorld.Object(blue,   yellow)
state_grid[0,  12] = ObjectWorld.Object(red,    blue)
state_grid[0,  13] = ObjectWorld.Object(blue,   grey)

state_grid[1,   6] = ObjectWorld.Object(blue,   cyan)
state_grid[1,  10] = ObjectWorld.Object(blue,   yellow)

state_grid[2,   4] = ObjectWorld.Object(grey,   red)
state_grid[2,   9] = ObjectWorld.Object(grey,   cyan)
state_grid[2,  13] = ObjectWorld.Object(cyan,   red)

state_grid[3,   1] = ObjectWorld.Object(red,    grey)
state_grid[3,   3] = ObjectWorld.Object(grey,   blue)
state_grid[3,   6] = ObjectWorld.Object(red,    green)
state_grid[3,   9] = ObjectWorld.Object(blue,   red)
state_grid[3,  14] = ObjectWorld.Object(green,  red)

state_grid[4,   4] = ObjectWorld.Object(cyan,   yellow)
state_grid[4,   9] = ObjectWorld.Object(cyan,   cyan)

state_grid[5,   2] = ObjectWorld.Object(pink,   cyan)
state_grid[5,   3] = ObjectWorld.Object(red,    grey)
state_grid[5,  15] = ObjectWorld.Object(yellow, grey)

state_grid[6,   0] = ObjectWorld.Object(blue,   blue)
state_grid[6,   1] = ObjectWorld.Object(red,    grey)
state_grid[6,   6] = ObjectWorld.Object(cyan,   grey)
state_grid[6,   7] = ObjectWorld.Object(red,    cyan)
state_grid[6,   8] = ObjectWorld.Object(green,  cyan)

state_grid[7,  11] = ObjectWorld.Object(cyan,   red)

state_grid[8,   3] = ObjectWorld.Object(grey,   blue)
state_grid[8,   6] = ObjectWorld.Object(green,  green)
state_grid[8,  15] = ObjectWorld.Object(green,  red)

state_grid[9,   6] = ObjectWorld.Object(blue,   blue)
state_grid[9,   7] = ObjectWorld.Object(red,    pink)
state_grid[9,  14] = ObjectWorld.Object(green,  pink)

state_grid[10,  0] = ObjectWorld.Object(grey,   blue)
state_grid[10,  7] = ObjectWorld.Object(yellow, red)
state_grid[10, 11] = ObjectWorld.Object(green,  yellow)

state_grid[11,  5] = ObjectWorld.Object(green,  blue)
state_grid[11, 11] = ObjectWorld.Object(cyan,   yellow)
state_grid[11, 13] = ObjectWorld.Object(green,  blue)

state_grid[12,  2] = ObjectWorld.Object(blue,   grey)
state_grid[12,  7] = ObjectWorld.Object(grey,   red)
state_grid[12, 13] = ObjectWorld.Object(cyan,   cyan)

state_grid[13,  9] = ObjectWorld.Object(pink,   blue)
state_grid[13, 11] = ObjectWorld.Object(pink,   cyan)
state_grid[13, 14] = ObjectWorld.Object(red,    yellow)
state_grid[13, 15] = ObjectWorld.Object(yellow, grey)

state_grid[14,  0] = ObjectWorld.Object(grey,   yellow)
state_grid[14, 10] = ObjectWorld.Object(yellow, red)
state_grid[14, 13] = ObjectWorld.Object(cyan,   blue)

state_grid[15,  0] = ObjectWorld.Object(red,    green)


ow = ObjectWorld(
    colors=colors,
    use_continuous_features=False,
    wrap_at_edges=False,
    use_manhattan_distance=False,
    state_grid=state_grid,
    name="Levine"
)

print(ow)
fig = ow.generate_figure()
ObjectWorld.save_figure(fig, "levine.pdf")
plt.show()
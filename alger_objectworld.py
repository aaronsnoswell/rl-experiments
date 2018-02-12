
import numpy as np
import matplotlib.pyplot as plt

from object_world import ObjectWorld


# Define some colors
red = ObjectWorld.Color('#800000')
green = ObjectWorld.Color('#008000')
blue = ObjectWorld.Color('#8080FF')
colors = [red, green, blue]

size = 10
state_grid = np.array([[None] * size for i in range(size)])

state_grid[0, 6] = ObjectWorld.Object(red,   red)
state_grid[0, 7] = ObjectWorld.Object(blue,  green)
state_grid[1, 8] = ObjectWorld.Object(blue,  green)
state_grid[2, 2] = ObjectWorld.Object(red,   green)
state_grid[3, 9] = ObjectWorld.Object(red,   green)
state_grid[4, 4] = ObjectWorld.Object(blue,  green)
state_grid[4, 5] = ObjectWorld.Object(green, green)
state_grid[5, 2] = ObjectWorld.Object(blue,  blue)
state_grid[5, 9] = ObjectWorld.Object(red,   green)
state_grid[6, 1] = ObjectWorld.Object(blue,  red)
state_grid[6, 3] = ObjectWorld.Object(red,   green)
state_grid[6, 6] = ObjectWorld.Object(green, red)
state_grid[8, 9] = ObjectWorld.Object(green, blue)
state_grid[9, 0] = ObjectWorld.Object(green, red)
state_grid[9, 4] = ObjectWorld.Object(blue,  blue)

ow = ObjectWorld(
    colors=colors,
    use_continuous_features=True,
    wrap_at_edges=False,
    use_manhattan_distance=True,
    state_grid=state_grid,
    name="Alger"
)

print(ow)
print(ow.feature_grid[:, :, 0])
fig = ow.generate_figure()
ObjectWorld.save_figure(fig, "alger.pdf")
plt.show()
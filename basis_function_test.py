
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def normal(mu, sigma, x):
    """
    1D Normal function
    """
    return math.exp(-1 * ((x - mu) ** 2) / (2 * sigma ** 2)) / \
        math.sqrt(2 * math.pi * sigma ** 2)


# Global variables
y_min = -1
y_max = 1


def main():

    # Build a set of basis functions
    d = 5
    sigma = 0.001
    min_pos = 0
    max_pos = 1
    step = (max_pos - min_pos) / d
    phi = [
        (lambda mu: lambda s: normal(mu, sigma, s))(p) for p in np.arange(
            min_pos + step/2,
            max_pos + step/2,
            step
        )
    ]

    # Precompute an alpha vector trajectory
    delta = 5
    N = 5000
    alpha_traj = np.zeros(shape=(N, d))
    alpha_traj[0, :] = np.random.uniform(size=d) * 2 - 1

    for i in range(1, N):

        alpha_traj[i, :] = alpha_traj[i-1, :] + \
            (np.random.uniform(size=d) * 2 - 1) * delta

        for ai in range(d):
            while alpha_traj[i, ai] > 1:
                alpha_traj[i, ai] -= 2
            while alpha_traj[i, ai] < -1:
                alpha_traj[i, ai] += 2

    # Specify the complete reward function
    R = lambda i, s: np.dot(alpha_traj[i, :], [phi[j](s) for j in range(len(phi))])

    # Initialize figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlim([min_pos, max_pos])
    plt.ylim([y_min, y_max])
    ax.grid()

    x = np.arange(0, 1, 0.01)

    R_line, = ax.plot([], [], '-', lw=2)


    def init():
        R_line.set_data([], [])
        return [R_line]


    def animate(i):
        global y_min, y_max

        y = [R(i, xi) for xi in x]
        R_line.set_data(
            x,
            y
        )

        y_min = min(y_min, min(y))
        y_max = max(y_max, max(y))
        plt.ylim([y_min, y_max])

        return [R_line]


    ani = animation.FuncAnimation(
        fig,
        animate,
        range(N),
        interval=1,
        blit=True,
        init_func=init
    )

    #ani.save('squiggle.mp4', fps=15)
    plt.show()


if __name__ == "__main__":
    main()


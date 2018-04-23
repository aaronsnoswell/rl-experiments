
import math
import time
import numpy as np
import matplotlib.pyplot as plt


def normal(mu, sigma, x):
    """
    1D Normal function
    """
    return math.exp(-1 * ((x - mu) ** 2) / (2 * sigma ** 2)) / \
        math.sqrt(2 * math.pi * sigma ** 2)


def main():

    # Build a set of basis functions
    d = 5
    sigma = 0.05
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

    alpha = np.random.uniform(size=d) * 2 - 1
    R = lambda s: np.dot(alpha, [phi[i](s) for i in range(len(phi))])
    
    plt.figure()
    x = np.arange(0, 1, 0.01)

    delta = 0.3
    for i in range(100):
        # Plot the basis functions and approximated function
        plt.clf()
        for i in range(len(phi)):
            plt.plot(x, [alpha[i] * phi[i](xi) for xi in x], 'r--')
        plt.plot(x, [R(xi) for xi in x])

        plt.xlim([min_pos, max_pos])
        plt.ylim([-2, 2])
        plt.grid()

        #plt.show(block=False)
        plt.pause(0.05)
        
        alpha += (np.random.uniform(size=d) * 2 - 1) * delta
        for ai in range(len(alpha)):
            if alpha[ai] > 1:
                alpha[ai] = 1
            elif alpha[ai] < -1:
                alpha[ai] = -1


if __name__ == "__main__":
    main()


"""
Demonstrates the Kalman Filter
From Thrun, 2005 Probabilistic Robotics, ch3.2, p42
"""

import math
import numpy as np
import matplotlib.pyplot as plt

# Get the bayes, MDP directories in our path
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bayes import FilterParams, kalman_filter


def plot_normal(mean, std_dev, x_range=[0, 30], y_range=[0, 0.4], num_pts=100, **kwargs):
    """
    Generates a figure for a normal distribution
    """

    def norm_pdf(x, mean, std_dev):
        """
        Standard Normal Distribution
        """
        u = (x - mean) / abs(std_dev)
        y = (1 / (math.sqrt(2 * math.pi) * abs(std_dev))) * math.exp(-u * u / 2)
        return y


    if type(std_dev) == np.matrixlib.defmatrix.matrix:
        std_dev = std_dev[0, 0]
    
    x = np.linspace(x_range[0], x_range[1], num=num_pts)
    y = [norm_pdf(xi, mean, std_dev) for xi in x]

    fig = plt.gcf()
    ax = plt.gca()
    plt.plot(x, y, **kwargs)

    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.grid(b=True)

    # To set axes equal
    #ax.set_aspect("equal", adjustable="box")

    # To disable axis labels
    #ax.tick_params(length=0, labelbottom="off", labelleft="off")


# Configure filter settings
params = FilterParams(
    np.matrix([1]),
    np.matrix([1]),
    np.matrix([1]),
    np.matrix([1]),
    np.matrix([2])
)

# Set initial conditions
initial_belief = (np.array([8]), np.matrix([4]))

# Arrays of actions and measurements
action_array = [np.array([0]), np.array([+15.5])]
measurement_array = [np.array([6]), np.array([25])]

belief = initial_belief
fig = plt.figure()

for t in range(len(action_array)):
    action = action_array[t]
    measurement = measurement_array[t]

    # Step once
    prediction, gain, new_belief = kalman_filter(
        belief,
        action,
        measurement,
        params
    )

    plot_normal(belief[0], belief[1], color='lightsteelblue', label='Prior')
    plot_normal(prediction[0], prediction[1], linestyle='dashed', color='red', label='Prediction')
    plot_normal(measurement, params.Q[0, 0], linestyle='dashed', color='grey', label='Measurement')
    plot_normal(new_belief[0], new_belief[1], color='royalblue', label='Posterior')
    plt.title("Belief state at t={}".format(t))
    plt.legend()
    plt.show()

    belief = new_belief


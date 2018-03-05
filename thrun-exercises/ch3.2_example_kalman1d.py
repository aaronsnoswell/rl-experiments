"""
Demonstrates the Kalman Filter
From Thrun, 2005 Probabilistic Robotics, ch3.2, p42
"""

import numpy as np
import math

import matplotlib as mpl
import matplotlib.pyplot as plt


def kalman_filter(mean, covariance, action, observation, A, B, R, C, Q):
    """
    The Kalman filter represents the belief by a mean and covariance

    @param mean        Mean vector at time t-1
    @param covariance  Covariance matrix at time t-1
    @param action      Action/control vector taken at time t-1
    @param observation Observation vector received after taking action u
    """

    # Check types of inputs
    assert type(mean) == numpy.ndarray,\
        "mean must be of type numpy.ndarray"
    assert type(covariance) == numpy.ndarray,\
        "covariance must be of type numpy.ndarray"
    assert type(action) == numpy.ndarray,\
        "action must be of type numpy.ndarray"
    assert type(observation) == numpy.ndarray,\
        "observation must be of type numpy.ndarray"

    assert type(A) == np.matrixlib.defmatrix.matrix,\
        "A must be of type numpy.matrix"
    assert type(B) == np.matrixlib.defmatrix.matrix,\
        "B must be of type numpy.matrix"
    assert type(R) == np.matrixlib.defmatrix.matrix,\
        "R must be of type numpy.matrix"
    assert type(C) == np.matrixlib.defmatrix.matrix,\
        "C must be of type numpy.matrix"
    assert type(Q) == np.matrixlib.defmatrix.matrix,\
        "Q must be of type numpy.matrix"

    # Calculate predicted belief belbar(x)
    mean_prediction = A * mean + B * u
    covariance_prediction = A * covariance * A.T + R

    # Compute Kalman gain
    K = covariance_prediction * C.T * np.linalg.inv(C * covariance_prediction * C.T + Q)

    mean = mean_prediction + K * (observation - C * mean_prediction)
    simga = (np.identity(K.shape[0]) - K * C) * covariance_prediction

    return mean, covariance


def norm_pdf(x, mean, std_dev):
    """
    Standard Normal Distribution
    """
    u = (x - mean) / abs(std_dev)
    y = (1 / (math.sqrt(2 * math.pi) * abs(std_dev))) * math.exp(-u * u / 2)
    return y


def plot_normal(mean, std_dev, x_range=[0, 30], y_range=[0, 0.25], **kwargs):
    """
    Generates a figure for a normal distribution
    """
    x = np.linspace(x_range[0], x_range[1])
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


mean = 8
covariance = 4

fig = plt.figure()
plot_normal(mean, covariance)
plot_normal(mean + 1, covariance*2)
plt.title("Normal distribution")
plt.show()


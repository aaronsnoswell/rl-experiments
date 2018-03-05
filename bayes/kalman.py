"""
Implements Kalman filtering methods
"""

import numpy as np
import math

import matplotlib as mpl
import matplotlib.pyplot as plt


class FilterParams():
    """
    Container class for a set of Kalman filter parameters
    """

    def __init__(self, A, B, R, C, Q):
        """
        Constructor

        @param A Controls how the current state affects the next state
        @param B Controls how the current action affects the next state
        
        @param R The covariance of the state transition noise

        @param C Controls how the current state affects the measurement probability
        @param Q The covariance of the measurement noise
        """

        assert type(A) == np.matrixlib.defmatrix.matrix,\
            "A must be of type numpy.matrix, was {}".format(type(A))
        assert type(B) == np.matrixlib.defmatrix.matrix,\
            "B must be of type numpy.matrix, was {}".format(type(B))
        assert type(R) == np.matrixlib.defmatrix.matrix,\
            "R must be of type numpy.matrix, was {}".format(type(R))
        assert type(C) == np.matrixlib.defmatrix.matrix,\
            "C must be of type numpy.matrix, was {}".format(type(C))
        assert type(Q) == np.matrixlib.defmatrix.matrix,\
            "Q must be of type numpy.matrix, was {}".format(type(Q))

        self.A = A
        self.B = B
        self.R = R
        self.C = C
        self.Q = Q


def kalman_filter(belief, action, observation, filter_params):
    """
    The Kalman filter represents the belief by a mean and covariance

    @param belief        Initial belief (mean, covariance) at time t-1
    @param action        Action/control vector taken at time t-1
    @param observation   Observation vector received after taking action u
    @param filter_params Set of Kalman filter parameters

    @returns (prediction, gain, belief) Tuple containing the predicted state,
        the kalman gain for this step, and the updated belief
    """

    # Extract belief parameters
    mean = belief[0]
    covariance = belief[1]

    # Unpack filter parameters for notaitonal convenience
    A = filter_params.A
    B = filter_params.B
    R = filter_params.R
    C = filter_params.C
    Q = filter_params.Q

    # Calculate predicted belief belbar(x)
    predicted_mean = A * mean + B * action
    predicted_covariance = A * covariance * A.T + R

    # Compute Kalman gain
    kalman_gain = predicted_covariance * C.T * np.linalg.inv(C * predicted_covariance * C.T + Q)

    mean = predicted_mean + kalman_gain * (observation - C * predicted_mean)
    covariance = (np.identity(kalman_gain.shape[0]) - kalman_gain * C) * predicted_covariance

    # Re-pack outputs to return prediction, gain, belief
    return (predicted_mean, predicted_covariance), kalman_gain, (mean, covariance)


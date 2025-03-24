import numpy as np
import constants as const
from scipy.linalg import solve_continuous_are


def A_nom(fe, th):
    """
    The time-independent A matrix for the linearized model.

    Parameters:
    ----------
    fe: float
        The thrust exerted by the missile under the nominal trajectory.
    th: float
        The pitch angle of the missile under the nominal trajectory.

    Returns:
    -------
    ndarray
    """
    m = const.MASS
    A = np.array([[0., 1., 0., 0., 0., 0.],
                  [0., 0., 0., 0., fe/m*np.cos(th), 0.],
                  [0., 0., 0., 1., 0., 0.],
                  [0., 0., 0., 0., -fe/m*np.sin(th), 0.],
                  [0., 0., 0., 0., 0., 1.],
                  [0., 0., 0., 0., 0., 0.]])
    return A


def B_nom(fe, th):
    """
    The time-independent B matrix for the linearized model.

    Parameters:
    ----------
    fe: float
        The thrust exerted by the missile under the nominal trajectory.
    th: float
        The pitch angle of the missile under the nominal trajectory.

    Returns:
    -------
    ndarray
    """
    l1 = const.L1
    l2 = const.L2
    ln = const.Ln
    m = const.MASS
    J = ((l1+l2)**2 + const.DIAMETER**2)/12*m

    B = np.array([[0., 0., 0.],
                  [1/m*np.sin(th), 1/m*np.cos(th), fe/m*np.cos(th)],
                  [0., 0., 0.],
                  [1/m*np.cos(th), -1/m*np.sin(th), -fe/m*np.sin(th)],
                  [0., 0., 0.],
                  [0., l2/J, -fe/J*(l1+ln)]])
    return B


def S(Q, R, fe, th):
    """
    Compute the S matrix which solves the Algebraic Riccati Equation for infinite-horizon
    LQR.

    Parameters:
    ----------
    Q: ndarray
        The state cost within the LQR objective.
    R: ndarray
        The input cost within the LQR objective.
    fe: float
        The thrust exerted by the missile under the nominal trajectory.
    th: float
        The pitch angle of the missile under the nominal trajectory.

    Returns:
    -------
    ndarray
    """
    A = A_nom(fe, th)
    B = B_nom(fe, th)
    return solve_continuous_are(A, B, Q, R)

import numpy as np
import src.constants as const


def f(s, u):
    """
    The dynamics function for the 2d missile.

    Parameters:
    ----------
    s : ndarray
        The state of the system.
    u : ndarray
        The input to the system.

    Returns:
    -------
    ndarray
        The time derivative of the state according to the system.
    """
    # Unravel state variables
    x, x_dot = s[0:2]
    z, z_dot = s[2:4]
    theta, theta_dot = s[4:6]
    # m = s[6]

    # Unravel control inputs
    f_e = u[0]
    f_s = u[1]
    phi = u[2]

    # Other constants.
    m = const.MASS
    inert_tensr = m  # TODO: Figure out the correct expression for the inertia tensor
    g = const.GRAVITY
    l1 = const.L1
    l2 = const.L2
    ln = const.Ln
    # alpha = const.ALPHA
    # beta = const.BETA

    s_dot = np.zeros(6)
    s_dot[0] = x_dot
    s_dot[1] = 1/m * (f_e * np.sin(phi + theta) + f_s * np.cos(theta))
    s_dot[2] = z_dot
    s_dot[3] = 1/m * (f_e * np.cos(phi + theta) - f_s * np.sin(theta) - m*g)
    s_dot[4] = theta_dot
    s_dot[5] = 1/inert_tensr * (l2 * f_s - f_e * np.sin(phi) * (l1 + np.cos(phi) * ln))
    # s_dot[6] = -alpha * f_e - beta * np.abs(f_s)

    return s_dot

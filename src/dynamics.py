import numpy as np


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
    # m = s[6] TODO: include mass dynamics later

    # Unravel control inputs
    f_e = u[0]
    f_s = u[1]
    phi = u[2]

    # Other constants.
    # TODO: Find better values for these constants.
    m = 1.  # TODO: include mass dynamics later
    inert_tensr = m
    g = 9.8
    l1 = 1.5
    l2 = 1.
    ln = 0.1
    alpha = 0.1
    beta = 0.1

    s_dot = np.zeros(7)
    s_dot[0] = x_dot
    s_dot[1] = 1/m * (f_e * np.sin(phi + theta) + f_s * np.cos(theta))
    s_dot[2] = z_dot
    s_dot[3] = 1/m * (f_e * np.cos(phi + theta) - f_s * np.sin(theta) - m*g)
    s_dot[4] = theta_dot
    s_dot[5] = 1/inert_tensr * (l2 * f_s - f_e * np.sin(phi) * (l1 + np.cos(phi) * ln))
    # s_dot[6] = -alpha * f_e - beta * np.abs(f_s)  TODO: include mass dynamics later

    return s_dot

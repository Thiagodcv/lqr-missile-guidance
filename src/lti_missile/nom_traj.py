import numpy as np
from scipy import optimize
import constants as const


def nom_traj_params(bc):
    """
    Solves for the parameters of the nominal trajectory given boundary conditions 'bc'.

    Parameters:
    ----------
    bc : Dict
        The dictionary containing boundary conditions for the nominal trajectory. Contains keys-values
        'x0' : x position at time 0.
        'x_dot0' : x velocity at time 0.
        'z0' : z position at time 0.
        'z_dot0' : z velocity at time 0.
        'T' : Amount of time until target reached.
        'xT' : x position at time T.
        'zT' : z position at time T.

    Returns:
    -------
    dict
        'fe' and 'theta' parameters of the nominal trajectory.
    """
    m = const.MASS
    g = const.GRAVITY
    T = bc['T']

    def fun(x):
        fe = x[0]
        th = x[1]
        return [1/(2*m)*fe*np.sin(th)*T**2 + bc['x_dot0']*T + bc['x0'] - bc['xT'],
                1/2*(1/m*fe*np.cos(th) - g)*T**2 + bc['z_dot0']*T + bc['z0'] - bc['zT']]

    def jac(x):
        fe = x[0]
        th = x[1]
        return T**2/(2*m) * np.array([[np.sin(th), fe*np.cos(th)],
                                      [np.cos(th), -fe*np.sin(th)]])

    x_init = [10_000, np.pi/4]
    sol = optimize.root(fun, x_init, jac=jac, method='hybr')
    return sol.x


def nom_state(t, fe, th, bc):
    """
    The state vector evaluated at time t along a nominal trajectory where the missile experiences thrust 'fe' and
    pitch angle 'th'.

    Parameters:
    ----------
    t: float
        The time.
    fe: float
        The thrust of the missile.
    th: float
        The pitch angle of the missile.
    bc: dict
        Defined in nom_traj_params().

    Returns:
    -------
    ndarray
    """
    m = const.MASS
    g = const.GRAVITY

    x = 1/(2*m)*fe*np.sin(th)*t**2 + bc['x_dot0']*t + bc['x0']
    x_dot = 1/m*fe*np.sin(th)*t + bc['x_dot0']
    z = 1/2*(1/m*fe*np.cos(th) - g)*t**2 + bc['z_dot0']*t + bc['z0']
    z_dot = (1/m*fe*np.cos(th) - g)*t + bc['z_dot0']
    th_dot = 0

    return np.array([x, x_dot, z, z_dot, th, th_dot])

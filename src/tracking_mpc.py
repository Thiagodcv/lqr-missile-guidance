import numpy as np
from scipy import optimize
import constants as const


class TrackingMPC:
    """
    An MPC controller which minimizes the distance between the missile and a reference
    trajectory at each time step t.
    """

    def __init__(self, Q, R, dt, N):
        """
        Parameters:
        ----------
        Q : ndarray
            Positive Semi-definite matrix for state reference cost.
        R : ndarray
            Positive definite matrix for input reference cost.
        dt : float
            The interval length between discrete consecutive timesteps.
        N : int
            The number of timesteps to optimize over.
        """
        self.Q = Q
        self.R = R
        self.dt = dt
        self.N = N

    def run(self, s_init, s_desired):
        pass


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
        'Fe' and 'theta' parameters of the nominal trajectory.
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

    x_init = [5, np.pi/4]
    sol = optimize.root(fun, x_init, jac=jac, method='hybr')
    return sol.x


def generate_nom_traj(bc, fe, th, dt):
    """
    Generate the reference trajectory for the missile.

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
    fe : float
        The thrust of the missile throughout its trajectory.
    th : float
        The angle of the missile with respect to global z axis (in radians).
    dt : float
        The interval length between discrete consecutive timesteps.

    Returns:
    -------
    ndarray
        An (N, 6)-array (where N = floor(T/dt)+1) containing the reference state
        at each timestep.
    """
    m = const.MASS
    g = const.GRAVITY
    T = bc['T']
    x = lambda t: 1/(2*m)*fe*np.sin(th)*t**2 + bc['x_dot0']*t + bc['x0'] - bc['xT']
    z = lambda t: 1/2*(1/m*fe*np.cos(th) - g)*t**2 + bc['z_dot0']*t + bc['z0'] - bc['zT']
    x_dot = lambda t: 1/m*fe*np.sin(th)*t + bc['x_dot0']
    z_dot = lambda t: (1/m*fe*np.cos(th) - g)*t + bc['z_dot0']

    N = np.floor(T/dt) + 1
    nom_traj = np.zeros((N, 5))
    t_vals = np.arange(0, T+dt, dt)

    nom_traj[:, 0] = x(t_vals)
    nom_traj[:, 1] = x_dot(t_vals)
    nom_traj[:, 2] = z(t_vals)
    nom_traj[:, 3] = z_dot(t_vals)
    nom_traj[:, 4] = th * np.ones(N)
    nom_traj[:, 5] = np.zeros(N)

    return nom_traj
    
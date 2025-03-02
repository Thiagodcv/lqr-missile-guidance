import numpy as np
import casadi as ca
from scipy import optimize
import constants as const


class TrackingMPC:
    """
    An MPC controller which minimizes the distance between the missile and a reference
    trajectory at each time step t.
    """

    def __init__(self, f, Q, R, dt, N, nom_s, nom_u):
        """
        Parameters:
        ----------
        f : function
            The dynamics function.
        Q : ndarray
            Positive Semi-definite matrix for state reference cost.
        R : ndarray
            Positive definite matrix for input reference cost.
        dt : float
            The interval length between discrete consecutive timesteps.
        N : int
            The number of timesteps to optimize over (including current state).
        nom_s : ndarray
            A (K+1, :) array containing the state reference trajectory.
        nom_u : ndarray
            A (K, :) array containing the input reference trajectory.
        """
        self.f = f
        self.Q = Q
        self.R = R
        self.dt = dt
        self.N = N
        self.nx = Q.shape[0]
        self.nu = R.shape[0]
        self.nom_s = nom_s
        self.nom_u = nom_u

        # The current index to start tracking the nominal trajectory from.
        self.curr_idx = 0

    def run(self, s_init, u_bounds=None, full=False):
        """
        Parameters:
        ----------
        s_init : ndarray
            The initial state.
        u_bounds : ndarray
            The dictionary containing upper and lower limits for the control inputs. With keys 'u_lb', 'u_ub'.
        """
        opti = ca.Opti()
        s = opti.variable(self.N+1, self.nx)
        u = opti.variable(self.N, self.nu)

        # Compute cost
        cost = 0
        N = min(self.nom_s.shape[0] - self.curr_idx - 1, self.N)
        for n in range(N+1):
            s_ref = self.nom_s[self.curr_idx + n, :]
            cost += ca.mtimes([(s[n, :] - s_ref).T, self.Q, (s[n, :] - s_ref)])
            if n < N:
                u_ref = self.nom_u[self.curr_idx + n, :]
                cost += ca.mtimes([(u[n, :] - u_ref).T, self.R, (u[n, :] - u_ref)])

        # Impose constraints
        if u_bounds is not None:
            u_lb = u_bounds["u_lb"]
            u_ub = u_bounds["u_ub"]
        opti.subject_to(s[0, :] == s_init)
        for n in range(N):
            opti.subject_to(s[n+1, :] == self.f(s[n, :], u[n, :]))
            if u_bounds is not None:
                opti.subject_to(u_lb <= u[n, :])
                opti.subject_to(u[n, :] <= u_ub)

        opti.solver("ipopt")
        sol = opti.solve()

        # Update nominal trajectory index
        self.curr_idx += 1

        if full:
            return sol.value(u)
        else:
            return sol.value(u[0, :])


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
    x = lambda t: 1/(2*m)*fe*np.sin(th)*t**2 + bc['x_dot0']*t + bc['x0']
    z = lambda t: 1/2*(1/m*fe*np.cos(th) - g)*t**2 + bc['z_dot0']*t + bc['z0']
    x_dot = lambda t: 1/m*fe*np.sin(th)*t + bc['x_dot0']
    z_dot = lambda t: (1/m*fe*np.cos(th) - g)*t + bc['z_dot0']

    N = int(np.floor(T/dt) + 1)
    nom_traj = np.zeros((N, 6))
    t_vals = np.arange(0, T+dt, dt)

    nom_traj[:, 0] = x(t_vals)
    nom_traj[:, 1] = x_dot(t_vals)
    nom_traj[:, 2] = z(t_vals)
    nom_traj[:, 3] = z_dot(t_vals)
    nom_traj[:, 4] = th * np.ones(N)
    nom_traj[:, 5] = np.zeros(N)

    inpt_traj = np.zeros((N, 3))
    inpt_traj[:, 0] = fe * np.ones(N)

    return nom_traj, inpt_traj

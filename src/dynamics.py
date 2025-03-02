import numpy as np
import src.constants as const


class MissileEnv:
    """
    A gym-style environment for simulating a 2d missile with simplified dynamics.
    """

    def __init__(self, init_state, target, dt, tol=1e-6):
        """
        Parameters
        ----------
        init_state : ndarray
            Initial state of the missile.
        target : ndarray
            The missile target in xz coordinates.
        dt : float
            The length of time between discrete simulated timesteps.
        tol : float
            Maximum distance between missile and target required for episode to end.
        """

        self.init_state = init_state
        self.curr_state = self.init_state
        self.target = target
        self.dt = dt
        self.t = 0  # The current time
        self.tol = tol

    def reset(self):
        self.curr_state = self.init_state
        self.t = 0
        return self.curr_state, self.t

    def step(self, u):
        # Unpack state and input vectors
        s = self.curr_state
        s_dot = f(s, u)
        self.t += self.dt
        self.curr_state = s + s_dot * self.dt

        curr_pos = np.array([self.curr_state[0], self.curr_state[1]])
        targ_hit = True if np.linalg.norm(curr_pos - self.target) < self.tol else False
        return self.curr_state, self.t, targ_hit


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

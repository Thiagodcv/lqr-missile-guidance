import numpy as np
from scipy.integrate import solve_ivp
from src.ltv_missile.dynamics import f
from src.ltv_missile.lqr import A_nom, B_nom, get_S_interp, S
from src.ltv_missile.nom_traj import nom_traj_params, nom_state
import constants as const
from src.utils import plot_dynamics
import sdeint
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import os


def experiment():
    result = simulate_target(cx=-300., cz=300., dx=10_000., dz=0., n_sec=60.)
    # x = result[:, 0]
    # y = result[:, 1]
    # plt.plot(x, y, linestyle='-')
    # plt.xlim(-10000, 10000)
    # plt.ylim(-2000, 7500)
    # plt.show()


def simulate_target(cx, cz, dx, dz, n_sec):
    """
    Simulates an airborne target following a parabolic trajectory.

    Parameters:
    ----------
    cx: float
        Initial velocity of the target along the x-axis.
    cz: float
        The initial velocity of the target along the z-axis.
    dx: float
        The initial position of the target along the x-axis.
    dz: float
        The initial position of the target along the z-axis.
    n_sec: float
        The number of seconds to simulate the environment for.

    Return:
    ------
    ndarray
        The state of the target at each timestep.
    """

    g = const.GRAVITY
    A_targ = np.array([[0., 0., 1., 0.],
                       [0., 0., 0., 1.],
                       [0., 0., 0., 0.],
                       [0., 0., -g/cx, 0.]])
    s_init = np.array([dx, dz, cx, cz])

    # Matrix for diffusion term
    sig_x = 10.
    sig_z = 1.
    B_targ = np.zeros((4, 4))
    B_targ[2, 2] = sig_x
    B_targ[3, 3] = sig_z

    def f(x, t):
        return A_targ @ x

    def g(x, t):
        return B_targ

    t_span = np.linspace(0., n_sec, int(n_sec*100))
    result = sdeint.itoint(f, g, s_init, t_span)
    return result


if __name__ == '__main__':
    experiment()

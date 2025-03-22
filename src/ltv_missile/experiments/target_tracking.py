import numpy as np
from scipy.integrate import solve_ivp
from src.ltv_missile.dynamics import f
from src.ltv_missile.lqr import A_nom, B_nom, get_S_interp, S
from src.ltv_missile.nom_traj import min_time_nom_moving_targ, nom_state
import constants as const
from src.utils import plot_dynamics
import sdeint
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import os


def experiment():
    # In seconds
    n_sec = 40
    track_strt_time = 10.
    update_lqr_freq = 0.2
    # In Hertz
    fps = 100
    result = simulate_target(cx=-300., cz=300., dx=10_000., dz=0., n_sec=n_sec, fps=fps)

    # Define state and input cost
    Q = np.identity(7)
    Q[0, 0] = 4.
    Q[1, 1] = 0.04
    Q[2, 2] = 4.
    Q[3, 3] = 0.04
    Q[4, 4] = 2500.
    Q[5, 5] = 25.
    Q[6, 6] = 0.

    R = np.identity(3)
    R[0, 0] = 0.04
    R[1, 1] = 100.
    R[2, 2] = 2500.

    R_inv = np.linalg.inv(R)
    n = Q.shape[0]

    def opt_u(x, t, fe_nom, th_nom, bc, nom_input, S_interp):
        # Compute LQR control input
        K = R_inv @ B_nom(t, fe_nom, th_nom).T @ S(t, S_interp, n)
        dx = x - nom_state(t, fe_nom, th_nom, bc)
        u = -K @ dx + nom_input
        return u

    def dyn(x, t, fe_nom, th_nom, bc, nom_input, S_interp):
        # Closed-loop system
        u = opt_u(x, t, fe_nom, th_nom, bc, nom_input, S_interp)
        x_dot = f(x, u)
        return x_dot

    G_mat = np.zeros((7, 7))
    # G_mat[5, 5] = 0.1

    def G(x, t):
        return G_mat

    th0 = np.pi / 8
    m0 = const.MASS
    init_state = np.array([0., 0., 0., 0., th0, 0., m0])
    fe_max = 5000.

    # Define array for saving state history of rocket
    state_history = np.tile(init_state, (int(track_strt_time * fps + 1), 1))
    idx_from_end = -1

    # Simulate
    init_guess = None
    missile_state = init_state
    for ts in range(int(track_strt_time/update_lqr_freq), int(n_sec/update_lqr_freq)):
        targ_state = result[ts*int(fps*update_lqr_freq), :]  # ts * update_lqr_freq = actual time

        missile_bc = {'x0': missile_state[0],
                      'x_dot0': missile_state[1],
                      'z0': missile_state[2],
                      'z_dot0': missile_state[3]}

        targ_bc = {'x0': targ_state[0],
                   'x_dot0': targ_state[2],
                   'z0': targ_state[1],
                   'z_dot0': targ_state[3]}

        nom_traj_params = min_time_nom_moving_targ(missile_bc, targ_bc, fe_max, init_guess=init_guess)
        fe_nom, th_nom, T_nom = nom_traj_params.x
        init_guess = np.array([fe_nom, th_nom, T_nom])
        nom_input = np.array([fe_nom, 0., 0.])
        S_interp = get_S_interp(Q, Q, R, fe_nom, th_nom, T_nom)

        print('sec: ', ts*update_lqr_freq)
        print('missile_state: ', missile_state)
        print('targ_state: ', targ_state)
        print('fe_nom: ', fe_nom)
        print('th_nom: ', th_nom)
        print('T_nom: ', T_nom)
        print('----------------')

        dyn_inner = lambda x, t: dyn(x, t, fe_nom, th_nom, missile_bc, nom_input, S_interp)
        t_span = np.linspace(0., update_lqr_freq, num=int(fps*update_lqr_freq + 1))
        sol = sdeint.itoint(dyn_inner, G, missile_state, t_span)
        missile_state = sol[-1, :]  # sol[-1, :] is state evaluated at endpoint

        # Save state history
        state_history = np.concatenate((state_history, sol[1:, :]), axis=0)

        # Evaluate termination condition
        terminate, dist, idx_from_end = terminate_cond(sol[:-1, :],
                                                       result[ts*int(fps*update_lqr_freq):(ts+1)*int(fps*update_lqr_freq), :])
        if terminate:
            print("Terminated with distance to target: {:.1f}m".format(dist))
            break

    print('state_history.shape: ', state_history.shape)
    print('result.shape: ', result.shape)

    x_m = state_history[:-idx_from_end, 0]
    y_m = state_history[:-idx_from_end, 2]
    episode_len = len(x_m)
    plt.plot(x_m, y_m, linestyle='-', color='blue')

    x_t = result[:episode_len, 0]
    y_t = result[:episode_len, 1]
    plt.plot(x_t, y_t, linestyle='-', color='orange')


    plt.xlim(-5000, 12000)
    plt.ylim(-2000, 7500)
    plt.show()


def terminate_cond(missile_states, targ_states, max_dist=5):
    num_idx = missile_states.shape[0]
    for i in range(num_idx):
        dist = np.linalg.norm(missile_states[i, [0, 2]] - targ_states[i, [0, 1]])
        if dist < max_dist:
            return True, dist, num_idx - i

    return False, -1, -1


def simulate_target(cx, cz, dx, dz, n_sec, fps=100):
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
    fps: int
        Number of time steps simulated per second.

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
    sig_z = 0.
    B_targ = np.zeros((4, 4))
    B_targ[2, 2] = sig_x
    B_targ[3, 3] = sig_z

    def f(x, t):
        return A_targ @ x

    def g(x, t):
        return B_targ

    t_span = np.linspace(0., n_sec, int(n_sec*fps + 1))
    result = sdeint.itoint(f, g, s_init, t_span)
    return result


if __name__ == '__main__':
    experiment()

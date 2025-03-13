import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from src.ltv_missile.dynamics import f
from src.ltv_missile.lqr import A_nom, B_nom, get_S_interp, S
from src.ltv_missile.nom_traj import nom_traj_params, nom_state
import constants as const


def experiment():

    # Set desired target and terminal time
    bc = {'x0': 0,
          'x_dot0': 0,
          'z0': 0,
          'z_dot0': 0,
          'T': 5,
          'xT': 1000,
          'zT': 500}
    fe_nom, th_nom = nom_traj_params(bc)
    th_nom = th_nom % (2*np.pi)
    nom_input = np.array([fe_nom, 0., 0.])
    print("fe_nom: ", fe_nom)
    print("th_nom: ", th_nom)
    # print(nom_state(5, fe_nom, th_nom, bc))

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

    # Get S matrix estimate based on cubic splines
    S_interp = get_S_interp(Q, Q, R, fe_nom, th_nom, bc['T'])

    def opt_u(t, x):
        # Compute LQR control input
        K = R_inv @ B_nom(t, fe_nom, th_nom).T @ S(t, S_interp, n)
        dx = x - nom_state(t, fe_nom, th_nom, bc)
        u = -K @ dx + nom_input
        return u

    def dyn(t, x):
        # Closed-loop system
        u = opt_u(t, x)
        x_dot = f(x, u)
        return x_dot

    th0 = np.pi/8
    m0 = const.MASS
    init_state = np.array([0., 0., 0., 0., th0, 0., m0])

    sol = solve_ivp(dyn, [0., bc['T']], init_state)
    print(sol)

    # Plot results
    t_seq = sol.t
    x_seq = [sol.y[:, t] for t in range(sol.y.shape[1])]
    nom_state_seq = np.array([nom_state(t, fe_nom, th_nom, bc) for t in t_seq]).T
    true_input_seq = np.array([opt_u(t, x) for t, x in zip(t_seq, x_seq)]).T
    plot_dynamics(t_seq, sol, nom_state_seq, true_input_seq, fe_nom)


def plot_dynamics(t_seq, sol, nom_state_seq, true_input_seq, fe_nom):
    num_plots = 7
    fig, axes = plt.subplots(num_plots, 1, figsize=(8, 10), sharex=True)

    # x coordinate
    x_ax = axes[0]
    x_ax.plot(t_seq, sol.y[0, :])
    x_ax.plot(t_seq, nom_state_seq[0, :])
    # x_ax.legend()
    x_ax.set_ylabel("x(t)")

    # z coordinate
    z_ax = axes[1]
    z_ax.plot(t_seq, sol.y[2, :])
    z_ax.plot(t_seq, nom_state_seq[2, :])
    # z_ax.legend()
    z_ax.set_ylabel("z(t)")

    # theta angle
    th_ax = axes[2]
    th_ax.plot(t_seq, sol.y[4, :])
    th_ax.plot(t_seq, nom_state_seq[4, :])
    # th_ax.legend()
    th_ax.set_ylabel("theta(t)")

    # mass m
    m_ax = axes[3]
    m_ax.plot(t_seq, sol.y[6, :])
    m_ax.plot(t_seq, nom_state_seq[6, :])
    # m_ax.legend()
    m_ax.set_ylabel("m(t)")

    # trust fe
    fe_ax = axes[4]
    fe_ax.plot(t_seq, true_input_seq[0, :])
    fe_ax.plot(t_seq, fe_nom * np.ones(len(t_seq)))
    fe_ax.set_ylim(80500, 80575)
    # fe_ax.legend()
    fe_ax.set_ylabel("fe(t)")

    # side thrust fs
    fs_ax = axes[5]
    fs_ax.plot(t_seq, true_input_seq[1, :])
    fs_ax.plot(t_seq, np.zeros(len(t_seq)))
    # fs_ax.legend()
    fs_ax.set_ylabel("fs(t)")

    # nozzle angle
    phi_ax = axes[6]
    phi_ax.plot(t_seq, true_input_seq[2, :])
    phi_ax.plot(t_seq, np.zeros(len(t_seq)))
    # phi_ax.legend()
    phi_ax.set_ylabel("phi(t)")

    axes[-1].set_xlabel("Time t")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    experiment()

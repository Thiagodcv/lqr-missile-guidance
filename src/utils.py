import numpy as np
from matplotlib import pyplot as plt


def plot_dynamics(t_seq, sol, nom_state_seq, true_input_seq, fe_nom,
                  include_mass=True, fe_lim=None):
    num_plots = 7 if include_mass else 6
    fig, axes = plt.subplots(num_plots, 1, figsize=(8, 10), sharex=True)

    # x coordinate
    x_ax = axes[0]
    x_ax.plot(t_seq, sol[:, 0])
    x_ax.plot(t_seq, nom_state_seq[:, 0])
    # x_ax.legend()
    x_ax.set_ylabel("x(t)")

    # z coordinate
    z_ax = axes[1]
    z_ax.plot(t_seq, sol[:, 2])
    z_ax.plot(t_seq, nom_state_seq[:, 2])
    # z_ax.legend()
    z_ax.set_ylabel("z(t)")

    # theta angle
    th_ax = axes[2]
    th_ax.plot(t_seq, sol[:, 4])
    th_ax.plot(t_seq, nom_state_seq[:, 4])
    # th_ax.legend()
    th_ax.set_ylabel("theta(t)")

    k = 3 if include_mass else 2
    if include_mass:
        # mass m
        m_ax = axes[k]
        m_ax.plot(t_seq, sol[:, 6])
        m_ax.plot(t_seq, nom_state_seq[:, 6])
        # m_ax.legend()
        m_ax.set_ylabel("m(t)")

    # trust fe
    fe_ax = axes[k+1]
    fe_ax.plot(t_seq, true_input_seq[:, 0])
    fe_ax.plot(t_seq, fe_nom * np.ones(len(t_seq)))
    if fe_lim is not None:
        fe_ax.set_ylim(fe_lim[0], fe_lim[1])
    # fe_ax.legend()
    fe_ax.set_ylabel("fe(t)")

    # side thrust fs
    fs_ax = axes[k+2]
    fs_ax.plot(t_seq, true_input_seq[:, 1])
    fs_ax.plot(t_seq, np.zeros(len(t_seq)))
    # fs_ax.legend()
    fs_ax.set_ylabel("fs(t)")

    # nozzle angle
    phi_ax = axes[k+3]
    phi_ax.plot(t_seq, true_input_seq[:, 2])
    phi_ax.plot(t_seq, np.zeros(len(t_seq)))
    # phi_ax.legend()
    phi_ax.set_ylabel("phi(t)")

    axes[-1].set_xlabel("Time t")
    plt.tight_layout()
    plt.show()

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import os
from datetime import datetime


def plot_dynamics(t_seq, sol, nom_state_seq, true_input_seq, fe_nom,
                  include_mass=True, fe_lim=None):
    num_plots = 7 if include_mass else 6
    fig, axes = plt.subplots(num_plots, 1, figsize=(8, 10), sharex=True)

    # x coordinate
    x_ax = axes[0]
    x_ax.plot(t_seq, sol[:, 0], label="True Trajectory")
    x_ax.plot(t_seq, nom_state_seq[:, 0], label="Nominal Trajectory")
    x_ax.legend()
    x_ax.set_ylabel("x")

    # z coordinate
    z_ax = axes[1]
    z_ax.plot(t_seq, sol[:, 2])
    z_ax.plot(t_seq, nom_state_seq[:, 2])
    # z_ax.legend()
    z_ax.set_ylabel("z")

    # theta angle
    th_ax = axes[2]
    th_ax.plot(t_seq, sol[:, 4])
    th_ax.plot(t_seq, nom_state_seq[:, 4])
    # th_ax.legend()
    th_ax.set_ylabel("theta")

    k = 3 if include_mass else 2
    if include_mass:
        # mass m
        m_ax = axes[k]
        m_ax.plot(t_seq, sol[:, 6])
        m_ax.plot(t_seq, nom_state_seq[:, 6])
        # m_ax.legend()
        m_ax.set_ylabel("m")

    # trust fe
    fe_ax = axes[k+1]
    fe_ax.plot(t_seq, true_input_seq[:, 0])
    fe_ax.plot(t_seq, fe_nom * np.ones(len(t_seq)))
    if fe_lim is not None:
        fe_ax.set_ylim(fe_lim[0], fe_lim[1])
    # fe_ax.legend()
    fe_ax.set_ylabel("F_E")

    # side thrust fs
    fs_ax = axes[k+2]
    fs_ax.plot(t_seq, true_input_seq[:, 1])
    fs_ax.plot(t_seq, np.zeros(len(t_seq)))
    # fs_ax.legend()
    fs_ax.set_ylabel("F_S")

    # nozzle angle
    phi_ax = axes[k+3]
    phi_ax.plot(t_seq, true_input_seq[:, 2])
    phi_ax.plot(t_seq, np.zeros(len(t_seq)))
    # phi_ax.legend()
    phi_ax.set_ylabel("phi")

    axes[-1].set_xlabel("Time t")
    plt.tight_layout()
    plt.show()


def export_multiple_to_mp4(pickle_data):
    # Find out which episode has the longest length
    num_pairs = len(pickle_data)
    max_ep_len = -1
    for ep in range(num_pairs):
        ep_len = len(pickle_data[ep]["missile_hist"][:, 0])
        max_ep_len = max(max_ep_len, ep_len)

    # Set all arrays to have this length
    end_buffer = 300
    for ep in range(num_pairs):
        ep_len = len(pickle_data[ep]["missile_hist"][:, 0])
        len_diff = max_ep_len - ep_len
        if len_diff > 0:
            added_len = len_diff + end_buffer
        else:
            added_len = end_buffer

        last_missile_state = pickle_data[ep]["missile_hist"][-1, :]
        last_missile_state_rep = np.tile(last_missile_state, (added_len, 1))
        pickle_data[ep]["missile_hist"] = np.concatenate((pickle_data[ep]["missile_hist"],
                                                          last_missile_state_rep), axis=0)

        last_targ_state = pickle_data[ep]["targ_hist"][-1, :]
        last_targ_state_rep = np.tile(last_targ_state, (added_len, 1))
        pickle_data[ep]["targ_hist"] = np.concatenate((pickle_data[ep]["targ_hist"],
                                                       last_targ_state_rep), axis=0)

    fig, ax = plt.subplots()
    ax.set_xlim(-5000, 12000)
    ax.set_ylim(-2000, 7500)

    missile_traj_ls = [ax.plot([], [], lw=1, color="blue")[0] for _ in range(num_pairs)]
    target_traj_ls = [ax.plot([], [], lw=1, color="orange")[0] for _ in range(num_pairs)]
    missile_traj_ls[0] = ax.plot([], [], lw=1, color="blue", label="Missile")[0]
    target_traj_ls[0] = ax.plot([], [], lw=1, color="orange", label="Target")[0]

    def init():
        for ep in range(num_pairs):
            missile_traj_ls[ep].set_data([], [])
            target_traj_ls[ep].set_data([], [])
        return tuple(missile_traj_ls) + tuple(target_traj_ls)

    def update(frame):
        for ep in range(num_pairs):
            xdata_m = pickle_data[ep]["missile_hist"][:frame + 1, 0]
            zdata_m = pickle_data[ep]["missile_hist"][:frame + 1, 2]
            missile_traj_ls[ep].set_data(xdata_m, zdata_m)

            xdata_t = pickle_data[ep]["targ_hist"][:frame + 1, 0]
            zdata_t = pickle_data[ep]["targ_hist"][:frame + 1, 1]
            target_traj_ls[ep].set_data(xdata_t, zdata_t)
        return tuple(missile_traj_ls) + tuple(target_traj_ls)

    # Some modifications to the graph
    ax.axhline(0, color='black', linestyle='--')
    ax.plot(0, 0, 'go', label='Launch Position')
    ax.legend()

    ani = animation.FuncAnimation(fig=fig,
                                  func=update,
                                  frames=max_ep_len+end_buffer,
                                  init_func=init,
                                  interval=1,
                                  blit=True)
    now = datetime.now()
    run_name = now.strftime("%Y_%m_%d_%H_%M_%S")
    fps = 100
    ani.save(os.getcwd() + "/media/combined_" + run_name + ".mp4", writer="ffmpeg", fps=int(2.5 * fps))
    plt.show()

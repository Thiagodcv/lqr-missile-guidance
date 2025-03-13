import numpy as np
from other.tracking_mpc import nom_traj_params, generate_nom_traj, TrackingMPC
from other.dynamics import MissileEnv, f_casadi


def tune_tracking_mpc():
    bc = {'x0': 0.,
          'x_dot0': 0.,
          'z0': 0.,
          'z_dot0': 0.,
          'T': 5.,
          'xT': 1000.,  # travel 1400m in one second
          'zT': 1000.}
    T = bc['T']

    fe, th = nom_traj_params(bc)
    th = th % (2*np.pi)  # Normalize angle to lie in [0, 2pi].
    dt = 1/12
    nom_s, nom_u = generate_nom_traj(bc, fe, th, dt)

    k = 100.
    Q = np.zeros((6, 6))
    Q[0, 0] = 100
    Q[1, 1] = 1
    Q[2, 2] = 100
    Q[3, 3] = 1
    Q[4, 4] = 400
    Q[5, 5] = 100
    Q = k * Q

    rho = 1.
    R = np.identity(3)
    R[0, 0] = 0.25
    R[1, 1] = 25  # 0.25
    R[2, 2] = 400
    R = rho * R

    N = 50
    u_bounds = {"u_lb": np.array([0, -10, -np.pi/4]),
                "u_ub": np.array([np.inf, 10, np.pi/4])}
    mpc = TrackingMPC(f=f_casadi, Q=Q, R=R, dt=dt, N=N, nom_s=nom_s, nom_u=nom_u, u_bounds=u_bounds)

    # Construct environment
    targ = np.array([bc['xT'], bc['zT']])
    # init_state = np.zeros(6)
    init_state = np.array([0., 0., 0., 0., 0.2, 0.])  # nom_s[:, 0]
    env = MissileEnv(init_state=init_state, target=targ, dt=dt)

    K = int(T/dt)
    t = 0
    state = init_state
    true_s = np.zeros(nom_s.shape)
    true_s[:, 0] = state
    for k in range(K):
        u = mpc.run(state)

        print("state: ", state)
        print("ref state: ", nom_s[:, k])
        print("t: ", t)
        print("k: ", k)
        print("u: ", u)
        print("ref u: ", nom_u[:, k])
        print("============================")

        state, t, targ_hit = env.step(u)
        true_s[:, k+1] = state

    ssd = compute_ssd(true_s, nom_s)
    print("ssd: ", ssd)


def compute_ssd(true_traj, nom_traj):
    """
    Compute sum of square distances between xz values from the actual trajectory
    and the nominal one.

    Parameters:
    ----------
    true_traj : np.array
        An (n_s, N+1) array containing the true trajectory of the system.
    nom_traj : np.array
        An (n_s, N+1) array containing the nominal (reference) trajectory of the system.
    """
    x_true = true_traj[0, :]
    z_true = true_traj[2, :]
    x_nom = nom_traj[0, :]
    z_nom = nom_traj[2, :]

    return np.sqrt(np.linalg.norm(x_true - x_nom)**2 + np.linalg.norm(z_true - z_nom)**2)


if __name__ == '__main__':
    tune_tracking_mpc()

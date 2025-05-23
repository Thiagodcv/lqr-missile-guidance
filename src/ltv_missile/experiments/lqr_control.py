import numpy as np
import sdeint
from src.ltv_missile.dynamics import f
from src.ltv_missile.lqr import A_nom, B_nom, get_S_interp, S
from src.ltv_missile.nom_traj import nom_traj_params, nom_state
import constants as const
from src.utils import plot_dynamics


def experiment():
    """
    Control a missile with decaying mass and with the presence of noise using finite-horizon LQR.
    """

    # Set desired target and terminal time
    bc = {'x0': 0,
          'x_dot0': 0,
          'z0': 0,
          'z_dot0': 0,
          'm0': const.MASS,
          'T': 30,
          'xT': 20_000,
          'zT': 10_000}
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
    S_interp = get_S_interp(Q, Q, R, fe_nom, th_nom, bc['T'], bc)

    # Define functions needed for SDE simulation
    def opt_u(x, t):
        # Compute LQR control input
        K = R_inv @ B_nom(t, fe_nom, th_nom, bc).T @ S(t, S_interp, n)
        dx = x - nom_state(t, fe_nom, th_nom, bc)
        u = -K @ dx + nom_input
        return u

    def dyn(x, t):
        # Closed-loop system
        u = opt_u(x, t)
        x_dot = f(x, u)
        return x_dot

    G_mat = np.zeros((7, 7))
    G_mat[5, 5] = 0.1

    def G(x, t):
        return G_mat

    # Set initial conditions and run simulation
    th0 = np.pi/8
    m0 = const.MASS
    init_state = np.array([0., 0., 0., 0., th0, 0., m0])
    t_span = np.linspace(0.0, bc['T'], 3000)
    sol = sdeint.itoint(dyn, G, init_state, t_span)
    print(sol)

    # Plot results
    x_seq = [sol[t, :] for t in range(sol.shape[0])]
    nom_state_seq = np.array([nom_state(t, fe_nom, th_nom, bc) for t in t_span])
    true_input_seq = np.array([opt_u(x, t) for x, t in zip(x_seq, t_span)])
    plot_dynamics(t_span, sol, nom_state_seq, true_input_seq, fe_nom, fe_lim=[4075, 4225])


if __name__ == '__main__':
    experiment()

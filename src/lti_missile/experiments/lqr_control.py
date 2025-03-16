import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from src.lti_missile.dynamics import f
from src.lti_missile.lqr import A_nom, B_nom, S
from src.lti_missile.nom_traj import nom_traj_params, nom_state
import constants as const
from src.utils import plot_dynamics
import sdeint


def experiment():

    # Set desired target and terminal time
    bc = {'x0': 0,
          'x_dot0': 0,
          'z0': 0,
          'z_dot0': 0,
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
    Q = np.identity(6)
    Q[0, 0] = 4.
    Q[1, 1] = 0.04
    Q[2, 2] = 4.
    Q[3, 3] = 0.04
    Q[4, 4] = 2500.
    Q[5, 5] = 25.

    R = np.identity(3)
    R[0, 0] = 0.04
    R[1, 1] = 100.
    R[2, 2] = 2500.

    R_inv = np.linalg.inv(R)
    n = Q.shape[0]

    A = A_nom(fe_nom, th_nom)
    B = B_nom(fe_nom, th_nom)
    S_mat = S(Q, R, fe_nom, th_nom)

    def opt_u(x, t):
        # Compute LQR control input
        K = R_inv @ B.T @ S_mat
        dx = x - nom_state(t, fe_nom, th_nom, bc)
        u = -K @ dx + nom_input
        return u

    def dyn(x, t):
        # Closed-loop system
        u = opt_u(x, t)
        x_dot = f(x, u)
        return x_dot

    G_mat = np.zeros((6, 6))
    G_mat[5, 5] = 0.1

    def G(x, t):
        return G_mat

    th0 = np.pi / 8
    init_state = np.array([0., 0., 0., 0., th0, 0.])
    t_span = np.linspace(0.0, bc['T'], 3000)
    sol = sdeint.itoint(dyn, G, init_state, t_span)
    # print(sol)

    # Plot results
    x_seq = [sol[t, :] for t in range(sol.shape[0])]
    nom_state_seq = np.array([nom_state(t, fe_nom, th_nom, bc) for t in t_span])
    true_input_seq = np.array([opt_u(x, t) for x, t in zip(x_seq, t_span)])
    plot_dynamics(t_span, sol, nom_state_seq, true_input_seq, fe_nom, include_mass=False, fe_lim=[5450, 5575])


if __name__ == '__main__':
    experiment()

import numpy as np
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
    # print(fe_nom)
    # print(th_nom)
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

    def dyn(t, x):
        # Compute LQR control input
        K = R_inv @ B_nom(t, fe_nom, th_nom).T @ S(t, S_interp, n)
        dx = x - nom_state(t, fe_nom, th_nom, bc)
        u = -K @ dx + nom_input

        x_dot = f(x, u)
        return x_dot

    th0 = np.pi/8
    m0 = const.MASS
    init_state = np.array([0., 0., 0., 0., th0, 0., m0])

    sol = solve_ivp(dyn, [0., bc['T']], init_state)
    print(sol)


if __name__ == '__main__':
    experiment()

import numpy as np
from src.ltv_missile.dynamics import f
from src.ltv_missile.lqr import A_nom, B_nom, get_S_interp, S
from src.ltv_missile.nom_traj import nom_traj_params, nom_state


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
    print(fe_nom)
    print(th_nom)
    print(nom_state(5, fe_nom, th_nom, bc))

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
        K = R_inv @ B_nom(t, fe_nom, th_nom).T @ S(t, S_interp, n)
        pass


if __name__ == '__main__':
    experiment()

import numpy as np
from src.ltv_missile.dynamics import f
from src.ltv_missile.lqr import A_nom, B_nom, get_S_interp, S
from src.ltv_missile.nom_traj import nom_traj_params, eval_nom_traj


def experiment():

    # Set desired target and terminal time
    bc = {'x0': 0,
          'x_dot0': 0,
          'z0': 0,
          'z_dot0': 0,
          'T': 5,
          'xT': 100,
          'zT': 100}
    fe_nom, th_nom = nom_traj_params(bc)
    th_nom = th_nom % (2*np.pi)
    print(fe_nom)
    print(th_nom)


if __name__ == '__main__':
    experiment()

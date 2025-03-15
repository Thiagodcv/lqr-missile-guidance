import numpy as np
from scipy.integrate import solve_ivp
from src.ltv_missile.dynamics import f
from src.ltv_missile.lqr import A_nom, B_nom, get_S_interp, S
from src.ltv_missile.nom_traj import nom_traj_params, nom_state
import constants as const
from src.utils import plot_dynamics


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

    def dyn(t, x):
        # Closed-loop system
        u = nom_input
        x_dot = f(x, u)

        # Add noise to theta_ddot
        x_dot[5] += np.random.normal(scale=0.1)
        return x_dot

    th0 = th_nom
    m0 = const.MASS
    init_state = np.array([0., 0., 0., 0., th0, 0., m0])

    sol = solve_ivp(dyn, [0., bc['T']], init_state, rtol=1e-3, atol=1e-6)
    print(sol)

    # Plot results
    t_seq = sol.t
    nom_state_seq = np.array([nom_state(t, fe_nom, th_nom, bc) for t in t_seq]).T
    true_input_seq = np.array([nom_input for t in t_seq]).T
    plot_dynamics(t_seq, sol, nom_state_seq, true_input_seq, fe_nom, fe_lim=[4600, 4750])


if __name__ == '__main__':
    experiment()

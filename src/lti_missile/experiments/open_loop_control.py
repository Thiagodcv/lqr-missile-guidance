import numpy as np
from src.lti_missile.dynamics import f
from src.lti_missile.nom_traj import nom_traj_params, nom_state
from src.utils import plot_dynamics
import sdeint


def experiment():
    """
    Control a missile with fixed mass and with the presence of noise using an open-loop control law
    (it's supposed to behave poorly).
    """

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

    # Define dynamics needed for SDE simulation
    def dyn(x, t):
        # Closed-loop system
        u = nom_input
        x_dot = f(x, u)
        return x_dot

    G_mat = np.zeros((6, 6))
    G_mat[5, 5] = 0.1

    def G(x, t):
        return G_mat

    # Set initial conditions and run simulation
    th0 = th_nom
    init_state = np.array([0., 0., 0., 0., th0, 0.])
    t_span = np.linspace(0.0, bc['T'], 3000)
    sol = sdeint.itoint(dyn, G, init_state, t_span)
    # print(sol)

    # Plot results
    nom_state_seq = np.array([nom_state(t, fe_nom, th_nom, bc) for t in t_span])
    true_input_seq = np.array([nom_input for t in t_span])
    plot_dynamics(t_span, sol, nom_state_seq, true_input_seq, fe_nom, include_mass=False, fe_lim=[5450, 5575])


if __name__ == '__main__':
    experiment()

from unittest import TestCase
import src.constants as const
from other.tracking_mpc import nom_traj_params, generate_nom_traj
from other.dynamics import MissileEnv
import numpy as np


class TestDynamics(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_dynamics_func(self):
        """
        Tests the dynamics function and the environment class.
        """
        bc = {'x0': 0.,
              'x_dot0': 0.,
              'z0': 0.,
              'z_dot0': 0.,
              'T': 1.,
              'xT': 1000.,  # travel 1400m in one second
              'zT': 1000.}

        m = const.MASS
        g = const.GRAVITY
        T = bc['T']

        fe, th = nom_traj_params(bc)
        dt = 0.01
        nom_traj, inpt_traj = generate_nom_traj(bc, fe, th, dt)

        # Construct environment
        targ = np.array([bc['xT'], bc['zT']])
        curr_state = nom_traj[:, 0]
        env = MissileEnv(init_state=curr_state, target=targ, dt=dt)

        N = int(T/dt)
        t = 0
        for n in range(N):
            print("curr_state: ", curr_state)
            print("ref state: ", nom_traj[:, n])
            print("t: ", t)
            print("------------")
            curr_state, t, targ_hit = env.step(inpt_traj[:, n])

        print("curr_state: ", curr_state)
        print("ref state: ", nom_traj[:, N])

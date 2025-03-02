from unittest import TestCase
import numpy as np
from src.tracking_mpc import nom_traj_params, generate_nom_traj
from src.constants import MASS, GRAVITY


class TestTrackingMPC(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_nom_traj_params(self):
        bc = {'x0': 0.,
              'x_dot0': 0.,
              'z0': 0.,
              'z_dot0': 0.,
              'T': 1.,
              'xT': 1000.,  # travel 1400m in one second
              'zT': 1000.}

        m = MASS
        g = GRAVITY
        T = bc['T']

        def fun(x):
            fe = x[0]
            th = x[1]
            return np.array([1/(2*m)*fe*np.sin(th)*T**2 + bc['x_dot0']*T + bc['x0'] - bc['xT'],
                             1/2*(1/m*fe*np.cos(th) - g)*T**2 + bc['z_dot0']*T + bc['z0'] - bc['zT']])

        def jac(x):
            fe = x[0]
            th = x[1]
            return T**2/(2*m) * np.array([[np.sin(th), fe * np.cos(th)],
                                          [np.cos(th), -fe * np.sin(th)]])

        epsilon = 1e-6
        x = np.array([2., 3.])
        dfun_dfe = (fun(x + np.array([epsilon, 0.])) - fun(x)) / epsilon
        dfun_dth = (fun(x + np.array([0., epsilon])) - fun(x)) / epsilon
        jac_est = np.concatenate((dfun_dfe[:, None], dfun_dth[:, None]), axis=1)

        tol = 1e-5
        # self.assertTrue(np.linalg.norm(jac_est - jac(x)) < tol)

        sol = nom_traj_params(bc)
        self.assertTrue(np.linalg.norm(fun(sol)) < tol)
        print("F_E: ", sol[0])
        print("th: ", sol[1])
        print(fun(sol))

    def test_generate_nom_traj(self):
        bc = {'x0': 0.,
              'x_dot0': 0.,
              'z0': 0.,
              'z_dot0': 0.,
              'T': 1.,
              'xT': 1000.,  # travel 1400m in one second
              'zT': 1000.}

        m = MASS
        g = GRAVITY
        T = bc['T']

        fe, th = nom_traj_params(bc)
        dt = 0.01
        nom_traj, inpt_traj = generate_nom_traj(bc, fe, th, dt)

        tol = 1e-6
        self.assertTrue(nom_traj.shape == (101, 6))
        self.assertTrue(np.abs(nom_traj[0, 0] - bc['x0']) < tol)
        self.assertTrue(np.abs(nom_traj[0, 1] - bc['x_dot0']) < tol)
        self.assertTrue(np.abs(nom_traj[0, 2] - bc['z0']) < tol)
        self.assertTrue(np.abs(nom_traj[0, 3] - bc['z_dot0']) < tol)
        self.assertTrue(np.abs(nom_traj[0, 4] - th) < tol)
        self.assertTrue(np.abs(nom_traj[0, 5] - 0.) < tol)
        self.assertTrue(np.abs(nom_traj[-1, 0] - bc['xT']) < tol)
        self.assertTrue(np.abs(nom_traj[-1, 2] - bc['zT']) < tol)
        self.assertTrue(np.abs(nom_traj[-1, 4] - th) < tol)
        self.assertTrue(np.abs(nom_traj[-1, 5] - 0.) < tol)
        print(nom_traj)

        self.assertTrue(inpt_traj.shape == (101, 3))
        self.assertTrue(np.linalg.norm(inpt_traj[:, 0] - fe*np.ones(101)) < tol)
        self.assertTrue(np.linalg.norm(inpt_traj[:, 1] - np.zeros(101)) < tol)
        self.assertTrue(np.linalg.norm(inpt_traj[:, 2] - np.zeros(101)) < tol)
        print(inpt_traj)

from unittest import TestCase
import numpy as np
from src.tracking_mpc import nom_traj_params
from src.constants import MASS, GRAVITY


class TestMPPI(TestCase):

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

from unittest import TestCase
import numpy as np
from ltv_missile.nom_traj import func, jac, nom_traj_params, nom_state


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
              'T': 3.,
              'xT': 100.,  # travel 1400m in one second
              'zT': 100.}

        fe = 100
        th = np.pi/4
        epsilon = 1e-6

        dfunc_dfe = (func(fe+epsilon, th, bc) - func(fe, th, bc))/epsilon
        dfunc_dth = (func(fe, th+epsilon, bc) - func(fe, th, bc))/epsilon
        true_jac = jac(fe, th, bc)
        est_jac = np.concatenate((dfunc_dfe[:, None], dfunc_dth[:, None]), axis=1)
        print(est_jac)
        print(true_jac)

        tol = 1e-5
        # TODO: this is now failing
        self.assertTrue(np.allclose(est_jac, true_jac, rtol=tol, atol=tol))

        fe, th = nom_traj_params(bc)
        print("fe: ", fe)
        print("th: ", th)

        nom_pos_T = nom_state(bc['T'], fe, th, bc)[[0, 2]]
        print("nom_pos_T: ", nom_pos_T)
        targ_pos_T = np.array([bc['xT'], bc['zT']])
        self.assertTrue(np.allclose(nom_pos_T, targ_pos_T, rtol=tol, atol=tol))

    def test_nom_state(self):
        bc = {'x0': 0,
              'x_dot0': 0,
              'z0': 0,
              'z_dot0': 0,
              'T': 5,
              'xT': 1000,
              'zT': 500}
        t = 2
        fe = 80525
        th = 1.01

        epsilon = 1e-7
        s1 = nom_state(t, fe, th, bc)
        s2 = nom_state(t+epsilon, fe, th, bc)

        x_dot_est = (s2[0] - s1[0])/epsilon
        z_dot_est = (s2[2] - s1[2]) / epsilon

        x_dot_true = s1[1]
        z_dot_true = s2[3]

        tol = 1e-3
        print(x_dot_true)
        print(x_dot_est)
        self.assertTrue(np.abs(x_dot_est - x_dot_true) < tol)
        self.assertTrue(np.abs(z_dot_est - z_dot_true) < tol)

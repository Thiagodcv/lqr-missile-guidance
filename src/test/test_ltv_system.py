from unittest import TestCase
import numpy as np
from src.ltv_nom_traj import func, jac, nom_traj_params, eval_nom_traj
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
        self.assertTrue(np.allclose(est_jac, true_jac, rtol=tol, atol=tol))

        fe, th = nom_traj_params(bc)
        print("fe: ", fe)
        print("th: ", th)

        nom_pos_T = eval_nom_traj(bc['T'], fe, th, bc)
        print("nom_pos_T: ", nom_pos_T)
        targ_pos_T = np.array([bc['xT'], bc['zT']])
        self.assertTrue(np.allclose(nom_pos_T, targ_pos_T, rtol=tol, atol=tol))

from unittest import TestCase
import numpy as np
from src.ltv_nom_traj import func, jac, nom_traj_params, eval_nom_traj
from src.ltv_lqr import A_nom, B_nom


class TestLTVLQR(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_A_nom_B_nom(self):
        t = 1.5
        fe = 64.13
        th = 0.61

        A = A_nom(t, fe, th)
        B = B_nom(t, fe, th)

        A_test = np.array([[0., 1., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 34.6038, 0., -15.9216],
                           [0., 0., 0., 1., 0., 0., 0.],
                           [0., 0., 0., 0., -24.1852, 0., -22.7803],
                           [0., 0., 0., 0., 0., 1., 0.],
                           [0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0.]])

        B_test = np.array([[0., 0., 0.],
                           [0.3771, 0.5396, 34.6038],
                           [0., 0., 0.],
                           [0.5396, -0.3771, -24.1852],
                           [0., 0., 0.],
                           [0., 0.6583,  -67.5486],
                           [-0.0050, 0., 0.]])

        tol = 1e-3
        self.assertTrue(np.allclose(A, A_test, atol=tol, rtol=tol))
        self.assertTrue(np.allclose(B, B_test, atol=tol, rtol=tol))

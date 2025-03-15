from unittest import TestCase
import numpy as np
from lti_missile.lqr import A_nom, B_nom, S


class TestLTILQR(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_A_nom_B_nom(self):
        """
        Test to ensure A_nom and B_nom return same values as MATLAB implementation.
        """
        fe = 5400
        th = np.pi/4

        A = A_nom(fe, th)
        B = B_nom(fe, th)

        A_test = np.array([[0., 1., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 38.1838, 0.],
                           [0., 0., 0., 1., 0., 0.],
                           [0., 0., 0., 0., -38.1838, 0.],
                           [0., 0., 0., 0., 0., 1.],
                           [0., 0., 0., 0., 0., 0.]])

        B_test = np.array([[0., 0., 0.],
                           [0.0071, 0.0071, 38.1838],
                           [0., 0., 0.],
                           [0.0071, -0.0071, -38.1838],
                           [0., 0., 0.],
                           [0., 0.0199, -114.8732]])

        tol = 1e-3
        self.assertTrue(np.allclose(A, A_test, atol=tol, rtol=tol))
        self.assertTrue(np.allclose(B, B_test, atol=tol, rtol=tol))

    def test_S(self):
        fe = 5400
        th = np.pi/4

        Q = np.identity(6)
        Q[0, 0] = 4.
        Q[1, 1] = 0.04
        Q[2, 2] = 4.
        Q[3, 3] = 0.04
        Q[4, 4] = 2500.
        Q[5, 5] = 25.

        R = np.identity(3)
        R[0, 0] = 0.04
        R[1, 1] = 100.
        R[2, 2] = 2500.
        R_inv = np.linalg.inv(R)

        S_mat = S(Q, R, fe, th)

        B = B_nom(fe, th)
        K = R_inv @ B.T @ S_mat
        print(K)

        K_test = np.array([[7.0711, 31.6307, 7.0711, 31.6307, 0., 0.],
                           [0.0005, 0.0005, -0.0005, -0.0005, 0.0101, 0.0013],
                           [-0.0283, -0.0322, 0.0283, 0.0322, -1.3849, -0.2052]])

        tol = 1e-4
        self.assertTrue(np.allclose(K, K_test, atol=tol, rtol=tol))

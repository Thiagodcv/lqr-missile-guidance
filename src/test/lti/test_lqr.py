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
        fe = 80000
        th = np.pi/4

        A = A_nom(fe, th)
        B = B_nom(fe, th)

        A_test = np.array([[0., 1., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 56.5685, 0.],
                           [0., 0., 0., 1., 0., 0.],
                           [0., 0., 0., 0., -56.5685, 0.],
                           [0., 0., 0., 0., 0., 1.],
                           [0., 0., 0., 0., 0., 0.]])

        B_test = np.array([[0., 0., 0.],
                           [0.0007, 0.0007, 56.5685],
                           [0., 0., 0.],
                           [0.0007, -0.0007, -56.5685],
                           [0., 0., 0.],
                           [0., 0.0010, -128.0000]])

        tol = 1e-3
        self.assertTrue(np.allclose(A, A_test, atol=tol, rtol=tol))
        self.assertTrue(np.allclose(B, B_test, atol=tol, rtol=tol))

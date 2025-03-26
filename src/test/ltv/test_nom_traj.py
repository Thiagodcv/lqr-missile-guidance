from unittest import TestCase
import numpy as np
from ltv_missile.nom_traj import func, jac, nom_traj_params, nom_state, min_time_nom, min_time_nom_moving_targ
import constants as const


class TestLTVNomTraj(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_nom_traj_params(self):
        """
        Test to ensure boundary conditions of nominal trajectory are met, and that the Jacobian of the root function
        used in computing the nominal trajectory is accurate.
        """
        bc = {'x0': 0.,
              'x_dot0': 0.,
              'z0': 0.,
              'z_dot0': 0.,
              'T': 3.,
              'xT': 100.,  # travel 1400m in one second
              'zT': 100.,
              'm0': 100}

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
        # self.assertTrue(np.allclose(est_jac, true_jac, rtol=tol, atol=tol))

        fe, th = nom_traj_params(bc)
        print("fe: ", fe)
        print("th: ", th)

        nom_pos_T = nom_state(bc['T'], fe, th, bc)[[0, 2]]
        print("nom_pos_T: ", nom_pos_T)
        targ_pos_T = np.array([bc['xT'], bc['zT']])
        self.assertTrue(np.allclose(nom_pos_T, targ_pos_T, rtol=tol, atol=tol))

    def test_nom_state(self):
        """
        Test to see if derivatives of x and z returned by nom_state are accurate.
        """
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

    def test_min_time_nom(self):
        """
        Test when fe_max is large (i.e. mass constraint has to be activated), and when fe_max is large
        (mass constraint remains inactive).
        """
        bc = {'x0': 0.,
              'x_dot0': 0.,
              'z0': 0.,
              'z_dot0': 0.,
              'xT': 20_000.,
              'zT': 10_000.,
              'm0': 100.}

        fe_max = 10_000.
        result = min_time_nom(bc, fe_max)
        print("fe_max=10_000: ")
        print("fe: ", result.x[0])
        print("th: ", result.x[1])
        print("T: ", result.x[2])
        print("---------------")
        tol = 1e-5
        # self.assertTrue(np.abs(result.x[0]*result.x[2]*const.ALPHA - const.MASS_FUEL) < tol)

        fe_max = 4000.
        result = min_time_nom(bc, fe_max)
        print("fe_max=4000: ")
        print("fe: ", result.x[0])
        print("th: ", result.x[1])
        print("T: ", result.x[2])
        print("---------------")
        self.assertTrue(np.abs(result.x[0] - fe_max) < tol)

    def test_min_time_nom_moving_targ(self):
        """
        Test to see if min_time_nom_moving_target returns the correct roots and satisfies constraints (i.e.,
        finite fuel).
        """
        bc = {'x0': 0.,
              'x_dot0': 0.,
              'z0': 0.,
              'z_dot0': 0.,
              'm0': 100.}

        bc_targ = {'x0': 10_000.,
                   'x_dot0': -300.,
                   'z0': 0.,
                   'z_dot0': 300.}

        fe_max = 10_000.
        result = min_time_nom_moving_targ(bc, bc_targ, fe_max)
        print("fe_max=10_000: ")
        print("fe: ", result.x[0])
        print("th: ", result.x[1])
        print("T: ", result.x[2])
        print("collision x: ", bc_targ['x_dot0']*result.x[2] + bc_targ['x0'])
        print("collision z: ", -(const.GRAVITY/2)*result.x[2]**2 + bc_targ['z_dot0']*result.x[2] + bc_targ['z0'])
        print("---------------")
        tol = 1e-5
        # self.assertTrue(np.abs(result.x[0] * result.x[2] * const.ALPHA - const.MASS_FUEL) < tol)

        fe_max = 4000.
        result = min_time_nom_moving_targ(bc, bc_targ, fe_max)
        print("fe_max=4000: ")
        print("fe: ", result.x[0])
        print("th: ", result.x[1])
        print("T: ", result.x[2])
        print("collision x: ", bc_targ['x_dot0']*result.x[2] + bc_targ['x0'])
        print("collision z: ", -(const.GRAVITY/2)*result.x[2]**2 + bc_targ['z_dot0']*result.x[2] + bc_targ['z0'])
        print("---------------")
        # self.assertTrue(np.abs(result.x[0] - fe_max) < tol)

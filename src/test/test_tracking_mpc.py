from unittest import TestCase
import numpy as np
from src.tracking_mpc import nom_traj_params, generate_nom_traj, TrackingMPC
from src.constants import MASS, GRAVITY
from src.dynamics import MissileEnv, f_casadi


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
        self.assertTrue(nom_traj.shape == (6, 101))
        self.assertTrue(np.abs(nom_traj[0, 0] - bc['x0']) < tol)
        self.assertTrue(np.abs(nom_traj[1, 0] - bc['x_dot0']) < tol)
        self.assertTrue(np.abs(nom_traj[2, 0] - bc['z0']) < tol)
        self.assertTrue(np.abs(nom_traj[3, 0] - bc['z_dot0']) < tol)
        self.assertTrue(np.abs(nom_traj[4, 0] - th) < tol)
        self.assertTrue(np.abs(nom_traj[5, 0] - 0.) < tol)
        self.assertTrue(np.abs(nom_traj[0, -1] - bc['xT']) < tol)
        self.assertTrue(np.abs(nom_traj[2, -1] - bc['zT']) < tol)
        self.assertTrue(np.abs(nom_traj[4, -1] - th) < tol)
        self.assertTrue(np.abs(nom_traj[5, -1] - 0.) < tol)
        print(nom_traj)

        self.assertTrue(inpt_traj.shape == (3, 101))
        self.assertTrue(np.linalg.norm(inpt_traj[0, :] - fe*np.ones(101)) < tol)
        self.assertTrue(np.linalg.norm(inpt_traj[1, :] - np.zeros(101)) < tol)
        self.assertTrue(np.linalg.norm(inpt_traj[2, :] - np.zeros(101)) < tol)
        print(inpt_traj)

    def test_run_tracking_mpc(self):
        def f(x, u):
            return x + u

        nom_s = np.array([0., 2., 3., 7., 12.])[:, None]
        nom_u = np.array([2., 1, 4, 5.])[:, None]

        Q = np.array([1.])
        R = np.array([1.])
        dt = 0.1
        N = 3

        mpc = TrackingMPC(f=f, Q=Q, R=R, dt=dt, N=N, nom_s=nom_s, nom_u=nom_u)

        # Run simulation
        state = np.array([0.])
        inpts = []
        for n in range(4):
            u = mpc.run(state, full=False)
            state = f(state, u)
            inpts.append(u)

        print("Optimal control inputs: ", inpts)

    def test_tracking_mpc_missile(self):
        """
        Tests tracking MPC applied to the 2D rocket.
        """
        bc = {'x0': 0.,
              'x_dot0': 0.,
              'z0': 0.,
              'z_dot0': 0.,
              'T': 1.,
              'xT': 1000.,  # travel 1400m in one second
              'zT': 1000.}
        T = bc['T']

        fe, th = nom_traj_params(bc)
        dt = 0.01
        nom_s, nom_u = generate_nom_traj(bc, fe, th, dt)

        Q = np.identity(6)
        R = np.identity(3)
        N = 10

        mpc = TrackingMPC(f=f_casadi, Q=Q, R=R, dt=dt, N=N, nom_s=nom_s, nom_u=nom_u)

        # Construct environment
        targ = np.array([bc['xT'], bc['zT']])
        # init_state = np.zeros(6)
        init_state = nom_s[:, 0]
        env = MissileEnv(init_state=init_state, target=targ, dt=dt)

        K = int(T / dt)
        t = 0
        state = init_state
        for k in range(K):
            u = mpc.run(state)

            print("state: ", state)
            print("ref state: ", nom_s[:, k])
            print("t: ", t)
            print("k: ", k)
            print("u: ", u)
            print("ref u: ", nom_u[:, k])
            print("============================")

            state, t, targ_hit = env.step(u)

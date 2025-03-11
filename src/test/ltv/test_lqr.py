from unittest import TestCase
import numpy as np
from ltv_missile.lqr import A_nom, B_nom, diff_riccati_eq, get_S_interp, S
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
import scipy.io as sio


class TestLTVLQR(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_A_nom_B_nom(self):
        """
        Test to ensure A_nom and B_nom return same values as MATLAB implementation.
        """
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

    def test_diff_riccati_eq(self):
        """
        Test diff_riccati_eq function to ensure no crashing.
        """
        Q = np.identity(7)
        Qf = np.identity(7)
        R = np.identity(3)
        fe = 60
        th = np.pi/4
        sol = diff_riccati_eq(Q, Qf, R, fe, th, T_final=3.)
        S_seq_inv = sol.y.T.reshape(-1, *Q.shape)
        # print(sol.t)
        # print(S_seq[-1, :, :])  # Last index corresponds to t=0

        # Put S(t) and t in correct order
        S_seq = S_seq_inv[::-1, :, :]
        t_seq = sol.t[::-1]

        tol = 1e-7
        K = len(sol.t)
        for idx in range(K):
            self.assertTrue(np.allclose(S_seq_inv[idx, :, :], S_seq[K-idx-1, :, :], atol=tol, rtol=tol))

        # Interpolate S solution using cubic splines
        n = Q.shape[0]
        interpolators = [[interp1d(t_seq, S_seq[:, i, j],
                                   kind='cubic', fill_value='extrapolate') for j in range(n)] for i in range(n)]

        # GRAPH S(t)
        t_seq_even = np.arange(0., 3., 0.01)  # Evenly spaced time steps
        fig, axes = plt.subplots(n, n, figsize=(3 * n, 3 * n))
        for i in range(n):
            for j in range(n):
                ax = axes[i, j]
                # Plot RK solution
                ax.plot(t_seq, S_seq[:, i, j], label=f'S[{i}, {j}]', color='red', linestyle=":")

                # Plot interpolation
                S_ij_smooth = [interpolators[i][j](t) for t in t_seq_even]
                ax.plot(t_seq_even, S_ij_smooth, color='blue')

                ax.grid(True)

        fig.supxlabel("Time")
        fig.supylabel("S_ij")
        plt.show()

        # Save S_seq and t_seq in MATLAB data file
        save_loc = "C:/Users/thiag/OneDrive/Desktop/MECH 509/Project/Missile Matlab Code/"
        sio.savemat(save_loc + "riccati_data.mat", {'S_seq': S_seq, 't_seq': t_seq})

    def test_get_S_cubic(self):
        """
        Test get_S_cubic function to ensure no crashing.

        TODO: Figure out issue with vertical lines.
        """
        Q = np.identity(7)
        Qf = np.identity(7)
        R = np.identity(3)
        fe = 60
        th = np.pi/4
        T_final = 3.
        n = Q.shape[0]

        interp = get_S_interp(Q, Qf, R, fe, th, T_final=T_final)

        # GRAPH S(t)
        t_seq_even = np.arange(0., T_final, 0.01)  # Evenly spaced time steps
        # S_seq = np.zeros((len(t_seq_even), n, n))
        # for t in t_seq_even:
        #     idx = int(100*t)
        #     S_seq[idx, :, :] = S(t, interp, n)

        fig, axes = plt.subplots(n, n, figsize=(3 * n, 3 * n))
        for i in range(n):
            for j in range(n):
                ax = axes[i, j]

                # Plot interpolation
                ax.plot(t_seq_even, [S(t, interp, n)[i, j] for t in t_seq_even])
                ax.grid(True)

        fig.supxlabel("Time")
        fig.supylabel("S_ij")
        plt.show()

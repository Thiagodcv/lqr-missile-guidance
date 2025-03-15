import numpy as np
import constants as const
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


def A_nom(t, fe, th):
    m = const.MASS - const.ALPHA * fe * t
    A = np.array([[0., 1., 0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., fe/m*np.cos(th), 0., -(fe/m**2)*np.sin(th)],
                  [0., 0., 0., 1., 0., 0., 0.],
                  [0., 0., 0., 0., -fe/m*np.sin(th), 0., -(fe/m**2)*np.cos(th)],
                  [0., 0., 0., 0., 0., 1., 0.],
                  [0., 0., 0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0., 0., 0.]])
    return A


def B_nom(t, fe, th):
    l1 = const.L1
    l2 = const.L2
    ln = const.Ln
    alpha = const.ALPHA
    m = const.MASS - alpha * fe * t
    J = ((l1+l2)**2 + const.DIAMETER**2)/12*m

    B = np.array([[0., 0., 0.],
                  [1/m*np.sin(th), 1/m*np.cos(th), fe/m*np.cos(th)],
                  [0., 0., 0.],
                  [1/m*np.cos(th), -1/m*np.sin(th), -fe/m*np.sin(th)],
                  [0., 0., 0.],
                  [0., l2/J, -fe/J*(l1+ln)],
                  [-alpha, 0., 0.]])
    return B


def F(t, S_flat, Q, R, fe, th):
    A = A_nom(t, fe, th)
    B = B_nom(t, fe, th)
    S = S_flat.reshape(A.shape)
    R_inv = np.linalg.inv(R)  # Assuming R is diagonal

    dS = -S @ A - A.T @ S + S @ B @ R_inv @ B.T @ S - Q
    return dS.flatten()


def diff_riccati_eq(Q, Qf, R, fe, th, T_final):
    # F_wrap = lambda t, S: F(t, S, Q, R, fe, th)
    S_final = Qf.flatten()

    T_init = 0
    sol = solve_ivp(F, [T_final, T_init], S_final, args=(Q, R, fe, th))
    return sol


def get_S_interp(Q, Qf, R, fe, th, T_final):
    # Get S(t) evaluated at finitely-many points in [0, T_final] via differential riccati equation
    sol = diff_riccati_eq(Q, Qf, R, fe, th, T_final)
    S_seq_inv = sol.y.T.reshape(-1, *Q.shape)

    # Put S(t) and t in correct order
    S_seq = S_seq_inv[::-1, :, :]
    t_seq = sol.t[::-1]

    # Interpolate S solution using cubic splines
    n = Q.shape[0]
    interp = [[interp1d(t_seq, S_seq[:, i, j],
                        kind='cubic', fill_value='extrapolate') for j in range(n)] for i in range(n)]
    return interp


def S(t, interp, n):
    return np.array([[interp[i][j](t) for j in range(n)] for i in range(n)])

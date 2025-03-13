import numpy as np
import constants as const
from scipy.linalg import solve_continuous_are


def A_nom(fe, th):
    m = const.MASS
    A = np.array([[0., 1., 0., 0., 0., 0.],
                  [0., 0., 0., 0., fe/m*np.cos(th), 0.],
                  [0., 0., 0., 1., 0., 0.],
                  [0., 0., 0., 0., -fe/m*np.sin(th), 0.],
                  [0., 0., 0., 0., 0., 1.],
                  [0., 0., 0., 0., 0., 0.]])
    return A


def B_nom(fe, th):
    l1 = const.L1
    l2 = const.L2
    ln = const.Ln
    m = const.MASS
    J = m

    B = np.array([[0., 0., 0.],
                  [1/m*np.sin(th), 1/m*np.cos(th), fe/m*np.cos(th)],
                  [0., 0., 0.],
                  [1/m*np.cos(th), -1/m*np.sin(th), -fe/m*np.sin(th)],
                  [0., 0., 0.],
                  [0., l2/J, -fe/J*(l1+ln)]])
    return B


def S(Q, R, fe, th):
    A = A_nom(fe, th)
    B = B_nom(fe, th)
    return solve_continuous_are(A, B, Q, R)

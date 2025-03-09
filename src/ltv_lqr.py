import numpy as np
from scipy import optimize
import constants as const


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
    J = m

    B = np.array([[0., 0., 0.],
                  [1/m*np.sin(th), 1/m*np.cos(th), fe/m*np.cos(th)],
                  [0., 0., 0.],
                  [1/m*np.cos(th), -1/m*np.sin(th), -fe/m*np.sin(th)],
                  [0., 0., 0.],
                  [0., l2/J, -fe/J*(l1+ln)],
                  [-alpha, 0., 0.]])
    return B

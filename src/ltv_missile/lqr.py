import numpy as np
import constants as const
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


def A_nom(t, fe, th, bc):
    """
    The time-varying A(t) matrix for the linearized model.

    Parameters:
    ----------
    t: float
        The time.
    fe: float
        The thrust exerted by the missile under the nominal trajectory.
    th: float
        The pitch angle of the missile under the nominal trajectory.
    bc : Dict
        The dictionary containing boundary conditions for the nominal trajectory. Contains keys-values
        'x_dot0' : x velocity at time 0.
        'z_dot0' : z velocity at time 0.
        'm0': mass at time 0.

    Returns:
    -------
    ndarray
    """
    alpha = const.ALPHA
    m0 = const.MASS if 'm0' not in bc else bc['m0']
    m = const.MASS - alpha * fe * t
    g = const.GRAVITY

    c_x = -m0/(alpha * fe) * (bc['x_dot0'] + np.sin(th)/alpha)
    c_z = -m0/(alpha * fe) * (bc['z_dot0'] + np.cos(th)/alpha - g*m0/(2*alpha*fe))
    x_dot = -np.sin(th)/alpha - c_x*(alpha*fe)/(m0 - alpha*fe*t)
    z_dot = -np.cos(th)/alpha + g/(2*alpha*fe)*(m0 - alpha*fe*t) - c_z*(alpha * fe)/(m0 - alpha*fe*t)

    A = np.array([[0., 1., 0., 0., 0., 0., 0.],
                  [0., alpha*fe/m, 0., 0., fe/m*np.cos(th), 0., -(fe/m**2)*(np.sin(th) + alpha*x_dot)],
                  [0., 0., 0., 1., 0., 0., 0.],
                  [0., 0., 0., alpha*fe/m, -fe/m*np.sin(th), 0., -(fe/m**2)*(np.cos(th) + alpha*z_dot)],
                  [0., 0., 0., 0., 0., 1., 0.],
                  [0., 0., 0., 0., 0., alpha*fe/m, 0.],
                  [0., 0., 0., 0., 0., 0., 0.]])
    return A


def B_nom(t, fe, th, bc):
    """
    The time-varying B(t) matrix for the linearized model.

    Parameters:
    ----------
    t: float
        The time.
    fe: float
        The thrust exerted by the missile under the nominal trajectory.
    th: float
        The pitch angle of the missile under the nominal trajectory.
    bc : Dict
        The dictionary containing boundary conditions for the nominal trajectory. Contains keys-values
        'x_dot0' : x velocity at time 0.
        'z_dot0' : z velocity at time 0.
        'm0': mass at time 0.

    Returns:
    -------
    ndarray
    """
    l1 = const.L1
    l2 = const.L2
    ln = const.Ln
    alpha = const.ALPHA

    m0 = const.MASS if 'm0' not in bc else bc['m0']
    m = m0 - alpha*fe*t
    g = const.GRAVITY

    c = ((l1+l2)**2 + const.DIAMETER**2)/12
    J = c*m

    c_x = -m0/(alpha*fe) * (bc['x_dot0'] + np.sin(th)/alpha)
    c_z = -m0/(alpha*fe) * (bc['z_dot0'] + np.cos(th)/alpha - g*m0/(2*alpha*fe))
    x_dot = -np.sin(th)/alpha - c_x*(alpha*fe)/(m0 - alpha*fe*t)
    z_dot = -np.cos(th)/alpha + g/(2*alpha*fe)*(m0 - alpha*fe*t) - c_z*(alpha*fe)/(m0 - alpha*fe*t)

    B = np.array([[0., 0., 0.],
                  [1/m*np.sin(th) + alpha/m*x_dot, 1/m*np.cos(th), fe/m*np.cos(th)],
                  [0., 0., 0.],
                  [1/m*np.cos(th) + alpha/m*z_dot, -1/m*np.sin(th), -fe/m*np.sin(th)],
                  [0., 0., 0.],
                  [0., l2/J, -fe/J*(l1+ln)],
                  [-alpha, 0., 0.]])
    return B


def F(t, S_flat, Q, R, fe, th, bc):
    """
    The time-derivative of the S matrix according to its differential Riccati equation.

    Parameters:
    ----------
    t: float
        The time.
    S_flat: ndarray
        The (flattened) S matrix at time t.
    Q: ndarray
        The state cost within the LQR objective.
    R: ndarray
        The input cost within the LQR objective.
    fe: float
        The thrust exerted by the missile under the nominal trajectory.
    th: float
        The pitch angle of the missile under the nominal trajectory.
    bc: Dict
        Definition given in A_nom() function.

    Returns:
    -------
    ndarray
    """
    A = A_nom(t, fe, th, bc)
    B = B_nom(t, fe, th, bc)
    S = S_flat.reshape(A.shape)
    R_inv = np.linalg.inv(R)  # Assuming R is diagonal

    dS = -S @ A - A.T @ S + S @ B @ R_inv @ B.T @ S - Q
    return dS.flatten()


def diff_riccati_eq(Q, Qf, R, fe, th, T_final, bc):
    """
    Solve the Differential Riccati Equation and get S(t) evaluated at discrete points.

    Parameters:
    ----------
    Q: ndarray
        The state cost within the LQR objective.
    Qf: ndarray
        The terminal state cost within the LQR objective.
    R: ndarray
        The input cost within the LQR objective.
    fe: float
        The thrust exerted by the missile under the nominal trajectory.
    th: float
        The pitch angle of the missile under the nominal trajectory.
    T_final: float
        The final time; we solve for S(t) for t in [0, T_final].
    bc: Dict
        Definition given in A_nom() function.

    Returns:
    -------
    Bunch object
    """
    # F_wrap = lambda t, S: F(t, S, Q, R, fe, th)
    S_final = Qf.flatten()

    T_init = 0
    sol = solve_ivp(F, [T_final, T_init], S_final, args=(Q, R, fe, th, bc))
    return sol


def get_S_interp(Q, Qf, R, fe, th, T_final, bc):
    """
    Solve the Differential Riccati Equation and approximate S(t) using cubic splines.

    Parameters:
    ----------
    Q: ndarray
        The state cost within the LQR objective.
    Qf: ndarray
        The terminal state cost within the LQR objective.
    R: ndarray
        The input cost within the LQR objective.
    fe: float
        The thrust exerted by the missile under the nominal trajectory.
    th: float
        The pitch angle of the missile under the nominal trajectory.
    T_final: float
        The final time; we solve for S(t) for t in [0, T_final].
    bc: Dict
        Definition given in A_nom() function.

    Returns:
    -------
    2D list of scipy.interpolate.interp1d objects
    """
    # Get S(t) evaluated at finitely-many points in [0, T_final] via differential riccati equation
    sol = diff_riccati_eq(Q, Qf, R, fe, th, T_final, bc)
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
    """
    Evaluate the S(t) matrix.

    Parameters:
    ----------
    t: float
        The time.
    interp: 2D list of scipy.interpolate.interp1d objects
        i.e., the n*n cubic splines which approximate S(t).
    n: int
        The dimension of the S(t) matrix.

    Returns:
    -------
    ndarray
    """
    return np.array([[interp[i][j](t) for j in range(n)] for i in range(n)])

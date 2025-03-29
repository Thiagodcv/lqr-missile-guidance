import numpy as np
from scipy import optimize
import constants as const


def nom_traj_params(bc):
    """
    Solves for the parameters of the nominal trajectory given boundary conditions 'bc'.

    Parameters:
    ----------
    bc : Dict
        The dictionary containing boundary conditions for the nominal trajectory. Contains keys-values
        'x0' : x position at time 0.
        'x_dot0' : x velocity at time 0.
        'z0' : z position at time 0.
        'z_dot0' : z velocity at time 0.
        'm0': mass at time 0.
        'T' : Amount of time until target reached.
        'xT' : x position at time T.
        'zT' : z position at time T.

    Returns:
    -------
    dict
        'fe' and 'theta' parameters of the nominal trajectory.
    """
    def func_wrap(x):
        fe = x[0]
        th = x[1]
        return func(fe, th, bc)

    def jac_wrap(x):
        fe = x[0]
        th = x[1]
        return jac(fe, th, bc)

    x_init = np.array([1000, np.pi/4])
    sol = optimize.root(func_wrap, x_init, jac=jac_wrap, method='hybr')
    return sol.x


def func(fe, th, bc):
    """
    The function used for root-finding in nom_traj_params().

    Parameters:
    ----------
    fe: float
        The thrust of the missile.
    th: float
        The pitch angle of the missile.
    bc: dict
        Defined in nom_traj_params().

    Returns:
    -------
    ndarray
    """
    m0 = const.MASS if 'm0' not in bc else bc['m0']
    g = const.GRAVITY
    alpha = const.ALPHA

    c_x = -m0/(alpha*fe)*(bc['x_dot0'] + np.sin(th)/alpha)
    d_x = bc['x0'] - m0*np.sin(th)/(alpha**2 * fe) - c_x*np.log(m0)
    c_z = -m0/(alpha*fe)*(bc['z_dot0'] + np.cos(th)/alpha - g*m0/(2*alpha*fe))
    d_z = bc['z0'] - m0*np.cos(th)/(alpha**2 * fe) + g*m0**2/(4*alpha**2 * fe**2) - c_z*np.log(m0)
    xT = bc['xT']
    zT = bc['zT']
    T = bc['T']

    # print("mT: ", m0 - alpha*fe*T)
    # gx = -(1/alpha)*np.sin(th)*(T-m0/(alpha*fe))*np.log(m0 - alpha*fe*T) + T/alpha*np.sin(th) + c_x*T + d_x - xT
    # gz = (-(1/alpha)*np.cos(th)*(T-m0/(alpha*fe))*np.log(m0 - alpha*fe*T) + T/alpha*np.cos(th) - (1/2)*g*T**2
    #       + c_z*T + d_z - zT)
    gx = np.sin(th)/(alpha**2 * fe)*(m0 - alpha*fe*T) + c_x*np.log(m0 - alpha*fe*T) + d_x - xT
    gz = (np.cos(th)/(alpha**2 * fe)*(m0 - alpha*fe*T) - g/(4*alpha**2 * fe**2)*(m0 - alpha*fe*T)**2 +
          c_z*np.log(m0 - alpha*fe*T) + d_z - zT)
    return np.array([gx, gz])


def jac(fe, th, bc):
    """
    TODO: Double check to see if working as expected.
    The Jacobian of the function used for root-finding in nom_traj_params().

    Parameters:
    ----------
    fe: float
        The thrust of the missile.
    th: float
        The pitch angle of the missile.
    bc: dict
        Defined in nom_traj_params().

    Returns:
    -------
    ndarray
    """
    m0 = const.MASS if 'm0' not in bc else bc['m0']
    alpha = const.ALPHA
    g = const.GRAVITY
    T = bc['T']

    # Coefficients
    c_x = -m0 / (alpha * fe) * (bc['x_dot0'] + np.sin(th) / alpha)
    d_x = bc['x0'] - m0 * np.sin(th) / (alpha ** 2 * fe) - c_x * np.log(m0)
    c_z = -m0 / (alpha * fe) * (bc['z_dot0'] + np.cos(th) / alpha - g * m0 / (2 * alpha * fe))
    d_z = bc['z0'] - m0 * np.cos(th) / (alpha ** 2 * fe) + g * m0 ** 2 / (4 * alpha ** 2 * fe ** 2) - c_z * np.log(m0)

    # Derivatives of cx
    dcx_dfe = m0/(alpha * fe**2)*(bc['x_dot0'] + np.sin(th)/alpha)
    dcx_dtheta = -m0*np.cos(th)/(alpha**2 * fe)

    # Derivatives of dx
    ddx_dfe = m0*np.sin(th)/(alpha**2 * fe**2) - dcx_dfe*np.log(m0)
    ddx_dtheta = -m0*np.cos(th)/(alpha**2 * fe) - dcx_dtheta*np.log(m0)

    # Derivatives of cz
    dcz_dfe = (m0/(alpha * fe**2)*(bc['z_dot0'] + np.cos(th)/alpha - g*m0/(2*alpha*fe)) -
               m0/(alpha*fe)*(g*m0/(2*alpha*fe**2)))
    dcz_dtheta = m0*np.sin(th)/(alpha**2 * fe)

    # Derivatives of dz
    ddz_dfe = m0*np.cos(th)/(alpha**2 * fe**2) - g*m0**2/(2*alpha**2 * fe**3) - dcz_dfe*np.log(m0)
    ddz_dtheta = m0*np.sin(th)/(alpha**2 * fe) - dcz_dtheta*np.log(m0)

    # Compute Jacobian
    dgx_dfe = (-np.sin(th)/(alpha**2 * fe**2)*(m0 - alpha*fe*T) + np.sin(th)/(alpha**2 * fe)*(-alpha*T) +
               dcx_dfe * np.log(m0 - alpha*fe*T) + c_x*(-alpha*T)/(m0-alpha*fe*T) + ddx_dfe)

    dgz_dfe1 = -np.cos(th)/(alpha**2 * fe**2)*(m0 - alpha*fe*T) + np.cos(th)/(alpha**2 * fe)*(-alpha*T)
    dgz_dfe2 = g/(2*alpha**2 * fe**3)*(m0 - alpha*fe*T)**2 - g/(2*alpha**2 * fe**2)*(m0 - alpha*fe*T)*(-alpha*T)
    dgz_dfe3 = dcz_dfe*np.log(m0 - alpha*fe*T) + c_z*(-alpha*T)/(m0 - alpha*fe*T) + ddz_dfe
    dgz_dfe = dgz_dfe1 + dgz_dfe2 + dgz_dfe3

    dgx_dth = np.cos(th)/(alpha**2 * fe)*(m0 - alpha*fe*T) + dcx_dtheta*np.log(m0 - alpha*fe*T) + ddx_dtheta
    dgz_dth = -np.sin(th)/(alpha**2 * fe)*(m0 - alpha*fe*T) + dcz_dtheta*np.log(m0 - alpha*fe*T) + ddz_dtheta

    return np.array([[dgx_dfe, dgx_dth],
                     [dgz_dfe, dgz_dth]])


def min_time_nom(bc, fe_max):
    """
    Solves for the parameters of the nominal trajectory with the minimum intercept time
    given boundary conditions 'bc'.

    Parameters:
    ----------
    bc : Dict
        A dictionary containing boundary conditions for the nominal trajectory. Contains keys-values
        'x0' : x position at time 0.
        'x_dot0' : x velocity at time 0.
        'z0' : z position at time 0.
        'z_dot0' : z velocity at time 0.
        'm0': mass at time 0.
        'xT' : x position at time T.
        'zT' : z position at time T.

    fe_max : float
        The maximum thrust of the missile. Used for constraints.

    Returns:
    -------
    ndarray
        'fe', 'theta', and 'T' parameters of the nominal trajectory (T is the intercept time).
    """
    m0 = const.MASS if 'm0' not in bc else bc['m0']
    m_fuel = m0 - const.MASS_DRY
    g = const.GRAVITY
    alpha = const.ALPHA

    def objective(var):
        _, _, T = var
        return T

    def x_hit_constraint(var):
        fe, th, T = var
        c_x = -m0 / (alpha * fe) * (bc['x_dot0'] + np.sin(th) / alpha)
        d_x = bc['x0'] - m0 * np.sin(th) / (alpha ** 2 * fe) - c_x * np.log(m0)
        xT = bc['xT']
        constr = np.sin(th)/(alpha**2 * fe)*(m0 - alpha*fe*T) + c_x*np.log(m0 - alpha*fe*T) + d_x - xT
        return constr

    def z_hit_constraint(var):
        fe, th, T = var
        c_z = -m0 / (alpha * fe) * (bc['z_dot0'] + np.cos(th) / alpha - g * m0 / (2 * alpha * fe))
        d_z = (bc['z0'] - m0 * np.cos(th)/(alpha**2 * fe) + g*m0**2/(4*alpha**2 * fe**2) -
               c_z*np.log(m0))
        zT = bc['zT']
        constr = (np.cos(th)/(alpha**2 * fe)*(m0 - alpha*fe*T) - g/(4*alpha**2 * fe**2)*(m0 - alpha*fe*T)**2 +
                  c_z*np.log(m0 - alpha*fe*T) + d_z - zT)
        return constr

    def fuel_constraint(var):
        fe, th, T = var
        return (1.0*m_fuel)/alpha - fe*T

    constraints = [{'type': 'eq', 'fun': lambda var: x_hit_constraint(var)},
                   {'type': 'eq', 'fun': lambda var: z_hit_constraint(var)},
                   {'type': 'ineq', 'fun': lambda var: fuel_constraint(var)}]
    bounds = [(0, fe_max), (-np.pi/2, np.pi/2), (0, None)]
    init_guess = np.array([4000., np.pi/4, 40.])
    result = optimize.minimize(objective, init_guess, method='SLSQP', constraints=constraints, bounds=bounds)
    return result


def min_time_nom_moving_targ(bc, bc_targ, fe_max, init_guess=None):
    """
    Solves for the parameters of the nominal trajectory with the minimum hit-to-kill time.
    This function assumes the airborne target is following a parabolic path.

    Parameters:
    ----------
    bc : Dict
        A dictionary containing boundary conditions for the nominal trajectory. Contains keys-values
        'x0' : x position at time 0.
        'x_dot0' : x velocity at time 0.
        'z0' : z position at time 0.
        'z_dot0' : z velocity at time 0.
        'm0': mass at time 0.

    bc_targ : Dict
        A dictionary containing boundary conditions for the airborne target. Contains keys-values
        'x0' : x position at time 0.
        'x_dot0' : x velocity at time 0.
        'z0' : z position at time 0.
        'z_dot0' : z velocity at time 0.

    fe_max : float
        The maximum thrust of the missile. Used for constraints.

    Returns:
    -------
    ndarray
        'Fe', 'theta', and 'T'.
    """
    m0 = const.MASS if 'm0' not in bc else bc['m0']
    m_fuel = m0 - const.MASS_DRY
    g = const.GRAVITY
    alpha = const.ALPHA

    # Params for target model
    c_x_targ = bc_targ['x_dot0']
    d_x_targ = bc_targ['x0']
    c_z_targ = bc_targ['z_dot0']
    d_z_targ = bc_targ['z0']

    def x_targ_pos(t):
        return c_x_targ*t + d_x_targ

    def z_targ_pos(t):
        return -(g/2)*t**2 + c_z_targ*t + d_z_targ

    def objective(var):
        _, _, T = var
        return T

    def x_hit_constraint(var):
        fe, th, T = var
        c_x = -m0/(alpha*fe) * (bc['x_dot0'] + np.sin(th)/alpha)
        d_x = bc['x0'] - m0*np.sin(th)/(alpha**2 * fe) - c_x * np.log(m0)
        constr = np.sin(th)/(alpha**2 * fe) * (m0 - alpha*fe*T) + c_x * np.log(m0 - alpha*fe*T) + d_x - x_targ_pos(T)
        return constr

    def z_hit_constraint(var):
        fe, th, T = var
        c_z = -m0/(alpha*fe) * (bc['z_dot0'] + np.cos(th)/alpha - g*m0/(2*alpha*fe))
        d_z = (bc['z0'] - m0*np.cos(th)/(alpha**2 * fe) + g*m0**2/(4*alpha**2 * fe**2) -
               c_z*np.log(m0))
        constr = (np.cos(th)/(alpha**2 * fe) * (m0 - alpha*fe*T) - g/(4*alpha**2 * fe**2) * (m0 - alpha*fe*T)**2 +
                  c_z*np.log(m0 - alpha*fe*T) + d_z - z_targ_pos(T))
        return constr

    def fuel_constraint(var):
        fe, th, T = var
        return (1.0*m_fuel)/alpha - fe*T

    constraints = [{'type': 'eq', 'fun': lambda var: x_hit_constraint(var)},
                   {'type': 'eq', 'fun': lambda var: z_hit_constraint(var)},
                   {'type': 'ineq', 'fun': lambda var: fuel_constraint(var)}]
    bounds = [(0, fe_max), (-np.pi / 2, np.pi / 2), (0, None)]
    if init_guess is None:
        init_guess = np.array([4000., np.pi / 4, 40.])
    result = optimize.minimize(objective, init_guess, method='SLSQP', constraints=constraints, bounds=bounds)
    return result


def nom_state(t, fe, th, bc):
    """
    The state vector evaluated at time t along a nominal trajectory where the missile experiences thrust 'fe' and
    pitch angle 'th'.

    Parameters:
    ----------
    t: float
        The time.
    fe: float
        The thrust of the missile.
    th: float
        The pitch angle of the missile.
    bc: dict
        Defined in nom_traj_params().

    Returns:
    -------
    ndarray
    """
    m0 = const.MASS if 'm0' not in bc else bc['m0']
    g = const.GRAVITY
    alpha = const.ALPHA

    c_x = -m0 / (alpha * fe) * (bc['x_dot0'] + np.sin(th) / alpha)
    d_x = bc['x0'] - m0 * np.sin(th) / (alpha ** 2 * fe) - c_x * np.log(m0)
    c_z = -m0 / (alpha * fe) * (bc['z_dot0'] + np.cos(th) / alpha - g * m0 / (2 * alpha * fe))
    d_z = bc['z0'] - m0 * np.cos(th) / (alpha ** 2 * fe) + g * m0 ** 2 / (4 * alpha ** 2 * fe ** 2) - c_z * np.log(m0)

    x = np.sin(th)/(alpha**2 * fe)*(m0 - alpha*fe*t) + c_x*np.log(m0 - alpha*fe*t) + d_x
    z = (np.cos(th)/(alpha**2 * fe)*(m0 - alpha*fe*t) - g/(4*alpha**2 * fe**2)*(m0 - alpha*fe*t)**2 +
         c_z*np.log(m0 - alpha*fe*t) + d_z)

    x_dot = -np.sin(th)/alpha - c_x*(alpha*fe)/(m0 - alpha*fe*t)
    z_dot = -np.cos(th)/alpha + g/(2*alpha*fe)*(m0 - alpha*fe*t) - c_z*(alpha*fe)/(m0 - alpha*fe*t)

    th_dot = 0
    m = m0 - alpha*fe*t
    return np.array([x, x_dot, z, z_dot, th, th_dot, m])

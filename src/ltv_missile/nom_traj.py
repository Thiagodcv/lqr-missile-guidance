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
        'T' : Amount of time until target reached.
        'xT' : x position at time T.
        'zT' : z position at time T.

    Returns:
    -------
    dict
        'Fe' and 'theta' parameters of the nominal trajectory.
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
    m0 = const.MASS
    g = const.GRAVITY
    alpha = const.ALPHA

    c_x = bc['x_dot0'] + (1/alpha)*np.log(m0)*np.sin(th)
    d_x = bc['x0'] - m0/(alpha**2 * fe)*np.log(m0)*np.sin(th)
    c_z = bc['z_dot0'] + (1/alpha)*np.log(m0)*np.cos(th)
    d_z = bc['z0'] - m0/(alpha**2 * fe)*np.log(m0)*np.cos(th)
    xT = bc['xT']
    zT = bc['zT']
    T = bc['T']

    # print("mT: ", m0 - alpha*fe*T)
    gx = -(1/alpha)*np.sin(th)*(T-m0/(alpha*fe))*np.log(m0 - alpha*fe*T) + T/alpha*np.sin(th) + c_x*T + d_x - xT
    gz = (-(1/alpha)*np.cos(th)*(T-m0/(alpha*fe))*np.log(m0 - alpha*fe*T) + T/alpha*np.cos(th) - (1/2)*g*T**2
          + c_z*T + d_z - zT)
    return np.array([gx, gz])


def jac(fe, th, bc):
    m0 = const.MASS
    alpha = const.ALPHA
    T = bc['T']

    # derivatives of cx
    dcx_dfe = 0
    dcx_dtheta = 1/alpha * np.log(m0) * np.cos(th)

    # derivatives of dx
    ddx_dfe = m0/(alpha**2 * fe**2) * np.log(m0) * np.sin(th)
    ddx_dtheta = -m0/(fe * alpha**2) * np.log(m0) * np.cos(th)

    # derivatives of cz
    dcz_dfe = 0
    dcz_dtheta = -1/alpha * np.log(m0) * np.sin(th)

    # derivatives of dz
    ddz_dfe = m0/(alpha**2 * fe**2) * np.log(m0) * np.cos(th)
    ddz_dtheta = m0/(alpha**2 * fe) * np.log(m0) * np.sin(th)

    prod_rule_fe = m0/(alpha*fe**2)*np.log(m0 - alpha*fe*T) - (T - m0/(alpha*fe))*(alpha*T)/(m0 - alpha*fe*T)
    prod_rule_th = (T - m0/(alpha*fe))*np.log(m0 - alpha*fe*T)

    dgx_dfe = -(1/alpha)*np.sin(th)*prod_rule_fe + ddx_dfe
    dgz_dfe = -(1/alpha)*np.cos(th)*prod_rule_fe + ddz_dfe

    dgx_dth = -(1/alpha)*np.cos(th)*prod_rule_th + T/alpha*np.cos(th) + dcx_dtheta*T + ddx_dtheta
    dgz_dth = 1/alpha*np.sin(th)*prod_rule_th - T/alpha*np.sin(th) + dcz_dtheta*T + ddz_dtheta

    return np.array([[dgx_dfe, dgx_dth],
                     [dgz_dfe, dgz_dth]])


def min_time_nom(bc, fe_max):
    """
    Solves for the parameters of the nominal trajectory with the minimum hit-to-kill time
    given boundary conditions 'bc'.

    Parameters:
    ----------
    bc : Dict
        A dictionary containing boundary conditions for the nominal trajectory. Contains keys-values
        'x0' : x position at time 0.
        'x_dot0' : x velocity at time 0.
        'z0' : z position at time 0.
        'z_dot0' : z velocity at time 0.
        'xT' : x position at time T.
        'zT' : z position at time T.

    fe_max : float
        The maximum thrust of the missile. Used for constraints.

    Returns:
    -------
    ndarray
        'Fe', 'theta', and 'T'.
    """
    m0 = const.MASS
    m_fuel = const.MASS_FUEL
    g = const.GRAVITY
    alpha = const.ALPHA

    def objective(var):
        _, _, T = var
        return T

    def x_hit_constraint(var):
        fe, th, T = var
        c_x = bc['x_dot0'] + (1 / alpha) * np.log(m0) * np.sin(th)
        d_x = bc['x0'] - m0 / (alpha ** 2 * fe) * np.log(m0) * np.sin(th)
        xT = bc['xT']
        constr = -(1/alpha)*np.sin(th)*(T-m0/(alpha*fe))*np.log(m0 - alpha*fe*T) + T/alpha*np.sin(th) + c_x*T + d_x - xT
        return constr

    def z_hit_constraint(var):
        fe, th, T = var
        c_z = bc['z_dot0'] + (1 / alpha) * np.log(m0) * np.cos(th)
        d_z = bc['z0'] - m0 / (alpha ** 2 * fe) * np.log(m0) * np.cos(th)
        zT = bc['zT']
        constr = (-(1/alpha)*np.cos(th)*(T-m0/(alpha*fe))*np.log(m0 - alpha*fe*T) + T/alpha*np.cos(th) -
                  (1/2)*g*T**2 + c_z*T + d_z - zT)
        return constr

    def fuel_constraint(var):
        fe, th, T = var
        return m_fuel/alpha - fe*T

    constraints = [{'type': 'eq', 'fun': lambda var: x_hit_constraint(var)},
                   {'type': 'eq', 'fun': lambda var: z_hit_constraint(var)},
                   {'type': 'ineq', 'fun': lambda var: fuel_constraint(var)}]
    bounds = [(0, fe_max), (-np.pi/2, np.pi/2), (0, None)]
    init_guess = np.array([4000., np.pi/4, 40.])
    result = optimize.minimize(objective, init_guess, method='SLSQP', constraints=constraints, bounds=bounds)
    return result


def min_time_nom_moving_targ(bc, bc_targ, fe_max):
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
    m0 = const.MASS
    m_fuel = const.MASS_FUEL
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
        c_x = bc['x_dot0'] + (1 / alpha) * np.log(m0) * np.sin(th)
        d_x = bc['x0'] - m0 / (alpha ** 2 * fe) * np.log(m0) * np.sin(th)
        constr = (-(1/alpha) * np.sin(th) * (T - m0/(alpha*fe)) * np.log(m0 - alpha*fe*T) + T/alpha*np.sin(th) +
                  c_x*T + d_x - x_targ_pos(T))
        return constr

    def z_hit_constraint(var):
        fe, th, T = var
        c_z = bc['z_dot0'] + (1 / alpha) * np.log(m0) * np.cos(th)
        d_z = bc['z0'] - m0 / (alpha ** 2 * fe) * np.log(m0) * np.cos(th)
        constr = (-(1/alpha) * np.cos(th) * (T - m0/(alpha*fe)) * np.log(m0 - alpha*fe*T) + T/alpha*np.cos(th) -
                  (1/2)*g*T**2 + c_z*T + d_z - z_targ_pos(T))
        return constr

    def fuel_constraint(var):
        fe, th, T = var
        return m_fuel / alpha - fe * T

    constraints = [{'type': 'eq', 'fun': lambda var: x_hit_constraint(var)},
                   {'type': 'eq', 'fun': lambda var: z_hit_constraint(var)},
                   {'type': 'ineq', 'fun': lambda var: fuel_constraint(var)}]
    bounds = [(0, fe_max), (-np.pi / 2, np.pi / 2), (0, None)]
    init_guess = np.array([4000., np.pi / 4, 40.])
    result = optimize.minimize(objective, init_guess, method='SLSQP', constraints=constraints, bounds=bounds)
    return result


def nom_state(t, fe, th, bc):
    m0 = const.MASS
    g = const.GRAVITY
    alpha = const.ALPHA

    c_x = bc['x_dot0'] + (1 / alpha) * np.log(m0) * np.sin(th)
    d_x = bc['x0'] - m0 / (alpha ** 2 * fe) * np.log(m0) * np.sin(th)
    c_z = bc['z_dot0'] + (1 / alpha) * np.log(m0) * np.cos(th)
    d_z = bc['z0'] - m0 / (alpha ** 2 * fe) * np.log(m0) * np.cos(th)

    x = -(1/alpha)*np.sin(th)*(t - m0/(alpha * fe))*np.log(m0 - alpha*fe*t) + t/alpha*np.sin(th) + c_x*t + d_x
    z = -(1/alpha)*np.cos(th)*(t - m0/(alpha * fe))*np.log(m0 - alpha*fe*t) + t/alpha*np.cos(th) - (1/2)*g*t** 2 + c_z*t + d_z

    x_dot = -(1/alpha)*np.log(m0 - alpha*fe*t)*np.sin(th) + c_x
    z_dot = -(1/alpha)*np.log(m0 - alpha*fe*t)*np.cos(th) - g*t + c_z

    th_dot = 0
    m = m0 - alpha*fe*t
    return np.array([x, x_dot, z, z_dot, th, th_dot, m])

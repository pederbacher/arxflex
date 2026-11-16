import numpy as np
import scipy.optimize as optim
import scipy.interpolate as si

from arxflex.compile_model import BAD_FIT
from .construct_bspline_basisfunctions import construct_bspline_basisfunctions
from .solar_gain import solar_gain
from .construct_design_matrix import construct_design_matrix
from arxflex.least_squares import least_squares
from ._finish import _finish


def valve_knots_and_values(valveknots, valvefunctionvalues, lower=-10.0, upper=10.0):
    """Full PCHIP knots/values from the optimisation parameters.

    Knots: lower, lower + cumsum(valveknots), upper  (valveknots are increments)
    Values: 0, valvefunctionvalues, 1
    """
    knots = np.concatenate(([lower], lower + np.cumsum(valveknots), [upper]))
    values = np.concatenate(([0.0], valvefunctionvalues, [1.0]))
    return knots, values


def pchip_valve(valveknots, valvefunctionvalues):
    """Monotone (PCHIP) valve function v(dT), dT = Tset - Ti.
    Outside [-10, 10] PCHIP extrapolates (as in notebook)."""
    return si.PchipInterpolator(*valve_knots_and_values(valveknots, valvefunctionvalues))


def fit_splines_and_thermostat_parameters_valvesplines(params, data, roomname, order,
                                                       basissplines=None, use_solar=True,
                                                       bad_fit=BAD_FIT):
    """params = [bspline coefficients (k+1), valve knot increments (N_valve-2),
                 valve function values (N_valve-2)]
    (no bspline coefficients if use_solar=False; basissplines is then not used)"""
    nb = basissplines.shape[1] if use_solar else 0
    nv = (len(params) - nb) // 2
    coeff = params[:nb]
    valveknots = params[nb:nb + nv]
    values = params[nb + nv:]
    knots, _ = valve_knots_and_values(valveknots, values)
    if np.any(np.diff(knots) <= 0):  # PCHIP needs increasing knots (guard added, not in notebook)
        return bad_fit
    solar = solar_gain(len(data), basissplines, coeff) if use_solar else None
    X, y = construct_design_matrix(order, data, roomname, pchip_valve(valveknots, values), solar,
                                   use_solar=use_solar)
    return least_squares(X, y)[1]


def construct_linear_inequality_constraints_valve(bspline_order, N_valve, length, use_solar=True):
    """Valve-spline model constraints, returns (A, ub, lb):
      row 0 : N_valve <= sum of valve knot increments <= length - 1
              (the last interior knot stays below the upper end point)
      rows 1.. : -1 <= f_i - f_{i+1} <= 0, i.e. valve function values non-decreasing
    use_solar=False : parameter vector without the bspline coefficients
    """
    nb = bspline_order + 1 if use_solar else 0
    nv = N_valve - 2
    ub = np.concatenate(([length - 1], np.zeros(nv - 1)))
    lb = np.concatenate(([N_valve], -np.ones(nv - 1)))
    row0 = np.concatenate((np.zeros(nb), np.ones(nv), np.zeros(nv)))
    D = np.eye(nv - 1, nv) - np.eye(nv - 1, nv, k=1)  # f_i - f_{i+1}
    rows = np.hstack((np.zeros((nv - 1, nb + nv)), D))
    return np.vstack((row0, rows)), ub, lb


def run_valve_spline_model(data, samples_per_day, Tfor, roomname="1.09", Nd=2, bspline_order=4,
                           N_valve=5, n_train=850, refit_theta_on_all_data=True, use_solar=True):
    """Monotone PCHIP valve function (notebook cells 86-94).

    samples_per_day : samples per day of the data (48 for the 30-min notebook data)
    Tfor            : assumed forward temperature [degC]
    The nonlinear parameters are fitted on the first n_train samples.
    refit_theta_on_all_data=True reproduces the notebook, which then estimated
    the linear parameters on ALL data, so the "prediction" after n_train is not
    fully out-of-sample. Use False for a clean train/test split.
    use_solar=False : no solar radiation input; the bspline coefficients are
    left out of the optimisation and bspline_order is not used.
    """
    nb = bspline_order + 1 if use_solar else 0
    basis = (construct_bspline_basisfunctions(samples_per_day, bspline_order)
             if use_solar else None)
    valveknots0 = np.diff(np.linspace(-10, 10, N_valve)[:-1])
    values0 = np.array([0, 0.1, 0.9]) if N_valve == 5 else np.linspace(0.1, 0.9, N_valve - 2)
    params0 = np.concatenate((0.6 * np.ones(nb), valveknots0, values0))
    bounds = ([(-10, 10)] * nb + [(0.05, 20 - N_valve)] * (N_valve - 2)
              + [(0, 1)] * (N_valve - 2))
    A, ub, lb = construct_linear_inequality_constraints_valve(bspline_order, N_valve, 20, use_solar)
    train = data.iloc[:n_train]
    res = optim.minimize(fit_splines_and_thermostat_parameters_valvesplines, params0, bounds=bounds,
                         args=(train, roomname, Nd, basis, use_solar), method="SLSQP",
                         constraints=optim.LinearConstraint(A, lb, ub), options={"maxiter": 500})
    coeff = res.x[:nb] if use_solar else None
    valveknots, values = res.x[nb:nb + N_valve - 2], res.x[nb + N_valve - 2:]
    valve = pchip_valve(valveknots, values)

    def gain(n):
        return solar_gain(n, basis, coeff) if use_solar else None

    fit_data = data if refit_theta_on_all_data else train
    X, y = construct_design_matrix(Nd, fit_data, roomname, valve, gain(len(fit_data)), use_solar=use_solar)
    theta, S = least_squares(X, y)
    return _finish(data, roomname, Nd, theta, S, valve, gain(len(data)), res, Tfor=Tfor,
                   n_train=n_train, use_solar=use_solar, basis=basis, bspline_coef=coeff,
                   valve_knots=valve_knots_and_values(valveknots, values))

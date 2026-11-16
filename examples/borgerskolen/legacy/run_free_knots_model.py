import numpy as np
import scipy.optimize as optim

from arxflex.compile_model import BAD_FIT
from .construct_bspline_basisfunctions import construct_bspline_basisfunctions
from .solar_gain import solar_gain
from .sigmoid_valve import sigmoid_valve
from .construct_design_matrix import construct_design_matrix
from arxflex.least_squares import least_squares
from ._finish import _finish
from .run_sigmoid_model import run_sigmoid_model


def fit_splines_knots_and_thermostat_parameters(params, data, roomname, order, samples_per_day,
                                                bad_fit=BAD_FIT):
    """params = [alpha, Toffset, bspline coefficients (k+1), knot increments (k-2)]"""
    k = (len(params) - 1) // 2  # bspline_order
    alpha, Toffset = params[:2]
    coeff = params[2:2 + k + 1]
    knots = params[2 + k + 1:]
    try:
        basis = construct_bspline_basisfunctions(samples_per_day, k, knot_params=knots)
    except ValueError:  # knots invalid for the spline fit (guard added, not in notebook)
        return bad_fit
    solar = solar_gain(len(data), basis, coeff)
    X, y = construct_design_matrix(order, data, roomname, sigmoid_valve(alpha, Toffset), solar)
    return least_squares(X, y)[1]


def construct_linear_inequality_constraints(bspline_order):
    """Free-knots model: the last interior knot (sum of knot increments) must lie in [5, 22] h.
    Returns (A, ub, lb) for scipy.optimize.LinearConstraint(A, lb, ub)."""
    nknots = bspline_order - 2
    A = np.concatenate((np.zeros((1, bspline_order + 1 + 2)), np.ones((1, nknots))), axis=1)
    return A, np.array([22]), np.array([5])


def run_free_knots_model(data, samples_per_day, Tfor, roomname="0.07", Nd=2, bspline_order=5,
                         use_solar=True):
    """Sigmoid valve, B-spline knot positions estimated too (notebook cell 71).

    samples_per_day : samples per day of the data (48 for the 30-min notebook data)
    Tfor            : assumed forward temperature [degC]
    use_solar=False : no solar radiation input. The B-spline knots only shape
                      the solar gain, so this is the same model as
                      run_sigmoid_model(..., use_solar=False), which is used.
    """
    if not use_solar:
        return run_sigmoid_model(data, samples_per_day, Tfor, roomname, Nd, bspline_order,
                                 bound=10, use_solar=False)
    knots0 = np.diff(np.linspace(0, 24, bspline_order)[:-1])
    params0 = np.concatenate(([1, 0], np.ones(bspline_order + 1), knots0))
    bounds = [(0.2, 5), (-10, 10)] + [(-10, 10)] * (bspline_order + 1) + [(1.5, 18)] * (bspline_order - 2)
    A, ub, lb = construct_linear_inequality_constraints(bspline_order)
    res = optim.minimize(fit_splines_knots_and_thermostat_parameters, params0, bounds=bounds,
                         args=(data, roomname, Nd, samples_per_day), method="SLSQP",
                         constraints=optim.LinearConstraint(A, lb, ub), options={"maxiter": 500})
    coeff = res.x[2:2 + bspline_order + 1]
    basis = construct_bspline_basisfunctions(samples_per_day, bspline_order,
                                             knot_params=res.x[-(bspline_order - 2):])
    valve = sigmoid_valve(*res.x[:2])
    solar = solar_gain(len(data), basis, coeff)
    X, y = construct_design_matrix(Nd, data, roomname, valve, solar)
    theta, S = least_squares(X, y)
    return _finish(data, roomname, Nd, theta, S, valve, solar, res, Tfor=Tfor, basis=basis,
                   bspline_coef=coeff)

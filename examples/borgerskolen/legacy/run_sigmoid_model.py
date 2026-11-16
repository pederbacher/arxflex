import numpy as np
import scipy.optimize as optim

from .construct_bspline_basisfunctions import construct_bspline_basisfunctions
from .solar_gain import solar_gain
from .sigmoid_valve import sigmoid_valve
from .construct_design_matrix import construct_design_matrix
from arxflex.least_squares import least_squares
from .fit_splines_and_thermostat_parameters import fit_splines_and_thermostat_parameters
from ._finish import _finish


def run_sigmoid_model(data, samples_per_day, Tfor, roomname="0.07", Nd=2, bspline_order=5,
                      bound=10, use_solar=True):
    """Sigmoid valve, fixed equidistant B-spline knots (notebook cells 27 and 70).

    samples_per_day : samples per day of the data (48 for the 30-min notebook data)
    Tfor            : assumed forward temperature [degC]
    use_solar=False : no solar radiation input; only alpha and Toffset are
                      optimised and bspline_order is not used.
    """
    n_coef = bspline_order + 1 if use_solar else 0
    basis = (construct_bspline_basisfunctions(samples_per_day, bspline_order)
             if use_solar else None)
    params0 = np.concatenate(([1, 0], np.ones(n_coef)))
    bounds = [(0.2, 5), (-bound, bound)] + [(-bound, bound)] * n_coef
    res = optim.minimize(fit_splines_and_thermostat_parameters, params0, bounds=bounds,
                         args=(data, roomname, Nd, basis, use_solar), method="SLSQP")
    valve = sigmoid_valve(*res.x[:2])
    bspline_coef = res.x[2:] if use_solar else None
    solar = solar_gain(len(data), basis, bspline_coef) if use_solar else None
    X, y = construct_design_matrix(Nd, data, roomname, valve, solar, use_solar=use_solar)
    theta, S = least_squares(X, y)
    return _finish(data, roomname, Nd, theta, S, valve, solar, res, Tfor=Tfor,
                   use_solar=use_solar, basis=basis, bspline_coef=bspline_coef)
